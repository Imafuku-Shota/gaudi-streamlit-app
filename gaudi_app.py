import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import time
import random
import itertools
import base64
import os
import requests
from PIL import Image

st.set_page_config(layout="centered")

# ============================================================
# 設定値
# ============================================================
NUM_ANCHORS = 8
anchors_x = np.linspace(-17.5, 17.5, NUM_ANCHORS)
anchors = [{"x": float(x), "y": 0.0} for x in anchors_x]

NUM_NEW_NODES = 41
NUM_INTERNAL_NODES = NUM_NEW_NODES - 2
MID_NODE_OFFSET = NUM_INTERNAL_NODES // 2

NUM_CHOICES = 4
INITIAL_STRINGS = 3
ADD_STRINGS_PER_ROUND = 1
ADDITION_ROUNDS = 4
FINAL_STRING_COUNT = INITIAL_STRINGS + ADD_STRINGS_PER_ROUND * ADDITION_ROUNDS

GRAVITY = 0.08
STIFFNESS = 0.95
DAMPING = 0.75

# ============================================================
# サイドバー：API設定とプロンプト
# ============================================================
st.sidebar.title("⚙️ AI生成設定")
st.sidebar.markdown("fal.ai APIキーが未入力の場合は、ダミー画像を表示します。")

fal_api_key = st.sidebar.text_input(
    "APIキー (fal.ai)",
    type="password",
    help="fal.ai のAPIキーを入力してください。Streamlit SecretsのFAL_KEYも使えます。"
)

flux_model = st.sidebar.selectbox(
    "FLUXモデル",
    [
        "fal-ai/flux-pro/kontext",
        "fal-ai/flux-pro/kontext/max"
    ],
    index=0
)

st.sidebar.markdown("---")

user_prompt = st.sidebar.text_area(
    "生成プロンプト",
    value="""Transform this structural skeleton into a completed Gaudi-inspired fantasy castle integrated into a dramatic landscape.

Use the input image as a structural guide for the main silhouette, arches, tower rhythm, and overall composition. Do not reproduce the black guide lines, dots, or flat graph-like appearance.

Preserve the main architectural skeleton, but reinterpret it as a realistic completed building.

Show the structure from a dynamic architectural viewpoint.

The castle should be part of a larger architectural scene with stone terraces, stairways, bridges, courtyards, trees, mountains, and atmospheric sky.

The structure should feel like a real inhabited building, not an isolated model. Add depth, perspective, shadows, surrounding ground, paths, small human figures, vegetation, and environmental context.

Organic Gaudi-inspired architecture, sculptural stone facade, flowing arches, spires, detailed windows, masonry texture, realistic architectural concept art, cinematic lighting, highly detailed.""",
    height=300
)

# ============================================================
# 基本関数
# ============================================================
def create_empty_structure():
    return {
        "nodes": [
            {"x": p["x"], "y": p["y"], "px": p["x"], "py": p["y"], "fixed": True}
            for p in anchors
        ],
        "links": [],
        "string_data": [],
        "added_pairs": []
    }


def deep_copy_structure(structure):
    return {
        "nodes": [dict(n) for n in structure["nodes"]],
        "links": list(structure["links"]),
        "string_data": [dict(s) for s in structure["string_data"]],
        "added_pairs": list(structure.get("added_pairs", []))
    }


def add_string_to_structure(structure, idx1, idx2):
    nodes = structure["nodes"]
    links = structure["links"]
    string_data = structure["string_data"]

    p1, p2 = nodes[idx1], nodes[idx2]
    dist = np.sqrt((p2["x"] - p1["x"])**2 + (p2["y"] - p1["y"])**2)

    if dist == 0:
        return False

    total_len = dist * 1.002
    seg_len = total_len / (NUM_NEW_NODES - 1)

    new_x = np.linspace(p1["x"], p2["x"], NUM_NEW_NODES)
    new_y = np.zeros(NUM_NEW_NODES)

    sag_depth = dist * 0.01
    for i in range(NUM_NEW_NODES):
        t = i / (NUM_NEW_NODES - 1)
        linear_y = p1["y"] + (p2["y"] - p1["y"]) * t
        sag = sag_depth * (4.0 * t * (1.0 - t))
        new_y[i] = linear_y - sag

    start_idx = len(nodes)
    first_link_idx = len(links)

    for i in range(1, NUM_NEW_NODES - 1):
        nodes.append({
            "x": float(new_x[i]),
            "y": float(new_y[i]),
            "px": float(new_x[i]),
            "py": float(new_y[i]),
            "fixed": False
        })

    prev_idx = idx1
    current_new_idx = start_idx

    for _ in range(1, NUM_NEW_NODES - 1):
        links.append((prev_idx, current_new_idx, seg_len))
        prev_idx = current_new_idx
        current_new_idx += 1

    links.append((prev_idx, idx2, seg_len))

    string_data.append({
        "id": len(string_data),
        "start_node": idx1,
        "end_node": idx2,
        "first_link_idx": first_link_idx,
        "last_link_idx": len(links) - 1
    })

    structure.setdefault("added_pairs", []).append((idx1, idx2))
    return True


def get_connection_candidates(structure):
    candidates = list(range(NUM_ANCHORS))

    for s in structure["string_data"]:
        base = NUM_ANCHORS + s["id"] * NUM_INTERNAL_NODES
        middle_node = base + MID_NODE_OFFSET
        if 0 <= middle_node < len(structure["nodes"]):
            candidates.append(middle_node)

    return candidates


def get_existing_pairs(structure):
    existing_pairs = set()
    for s in structure["string_data"]:
        pair = tuple(sorted((s["start_node"], s["end_node"])))
        existing_pairs.add(pair)
    return existing_pairs


def choose_random_pair(structure, rng):
    candidates = get_connection_candidates(structure)
    if len(candidates) < 2:
        return None

    existing_pairs = get_existing_pairs(structure)
    all_pairs = list(itertools.combinations(candidates, 2))

    MIN_HORIZONTAL_DISTANCE = 3.0
    valid_pairs = []

    for p in all_pairs:
        if tuple(sorted(p)) in existing_pairs:
            continue

        n1 = structure["nodes"][p[0]]
        n2 = structure["nodes"][p[1]]
        dx = abs(n2["x"] - n1["x"])

        if dx < MIN_HORIZONTAL_DISTANCE:
            continue

        valid_pairs.append(p)

    if not valid_pairs:
        return None

    return rng.choice(valid_pairs)


def add_random_strings(structure, num_strings, rng):
    added = 0

    for _ in range(num_strings):
        pair = choose_random_pair(structure, rng)
        if pair is None:
            break

        ok = add_string_to_structure(structure, pair[0], pair[1])
        if ok:
            added += 1

    return added


def simulate_structure(structure, steps=350, constraint_iterations=6):
    nodes = structure["nodes"]
    links = structure["links"]

    for _ in range(steps):
        for n in nodes:
            if not n.get("fixed", False):
                vx = (n["x"] - n["px"]) * DAMPING
                vy = (n["y"] - n["py"]) * DAMPING

                n["px"], n["py"] = n["x"], n["y"]
                n["x"] += vx
                n["y"] += vy - GRAVITY

        for _ in range(constraint_iterations):
            for idx1, idx2, target_dist in links:
                n1, n2 = nodes[idx1], nodes[idx2]

                dx = n2["x"] - n1["x"]
                dy = n2["y"] - n1["y"]
                d = np.sqrt(dx**2 + dy**2)

                if d == 0:
                    continue

                diff = (target_dist - d) / d * 0.5

                if not n1.get("fixed", False):
                    n1["x"] -= dx * diff * STIFFNESS
                    n1["y"] -= dy * diff * STIFFNESS

                if not n2.get("fixed", False):
                    n2["x"] += dx * diff * STIFFNESS
                    n2["y"] += dy * diff * STIFFNESS


def relax_structure_copy(structure):
    return deep_copy_structure(structure)


def get_active_bounds(structure):
    nodes = structure["nodes"]

    active_node_indices = set(range(NUM_ANCHORS))
    for s in structure["string_data"]:
        base = NUM_ANCHORS + s["id"] * NUM_INTERNAL_NODES
        active_node_indices.update(range(base, base + NUM_INTERNAL_NODES))

    ys = [nodes[i]["y"] for i in active_node_indices if 0 <= i < len(nodes)]
    min_y = min(ys) if ys else -15.0

    bottom_limit = min_y - 5.0
    half_width = max(20.0, abs(bottom_limit) * 0.8)

    return bottom_limit, half_width


def draw_structure(structure, inverted=False, small=False, highlight_new=False):
    fig_size = (3.2, 3.2) if small else (8, 8)
    dpi = 90 if small else 150
    line_width = 2.0 if small else 4.5
    anchor_size = 35 if small else 120

    fig, ax = plt.subplots(figsize=fig_size)
    nodes = structure["nodes"]

    ax.scatter(
        [p["x"] for p in anchors],
        [0] * NUM_ANCHORS,
        color="#555555",
        s=anchor_size,
        zorder=10
    )

    ax.plot(
        [-20, 20],
        [0, 0],
        color="#777777",
        lw=2 if small else 4,
        zorder=5
    )

    parent_string_count = structure.get("parent_string_count", None)

    for s in structure["string_data"]:
        base = NUM_ANCHORS + s["id"] * NUM_INTERNAL_NODES
        node_indices = [s["start_node"]] + list(range(base, base + NUM_INTERNAL_NODES)) + [s["end_node"]]

        xs = [nodes[i]["x"] for i in node_indices if 0 <= i < len(nodes)]
        ys = [nodes[i]["y"] for i in node_indices if 0 <= i < len(nodes)]

        is_new_string = (
            highlight_new
            and parent_string_count is not None
            and s["id"] >= parent_string_count
        )

        color = "#1C83E1" if is_new_string else "black"
        lw = line_width + 1.0 if is_new_string else line_width
        z_order = 8 if is_new_string else 6

        ax.plot(
            xs,
            ys,
            color=color,
            lw=lw,
            solid_joinstyle="round",
            solid_capstyle="round",
            zorder=z_order
        )

    bottom_limit, half_width = get_active_bounds(structure)

    if inverted:
        ax.set_ylim(5, bottom_limit)
    else:
        ax.set_ylim(bottom_limit, 5)

    ax.set_xlim(-half_width, half_width)
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def make_ai_input_image(structure):
    image_bytes = draw_structure(structure, inverted=True, small=False)
    raw_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    resized_img = raw_img.resize((1024, 1024), Image.LANCZOS)

    out_buf = io.BytesIO()
    resized_img.save(out_buf, format="PNG")
    return out_buf.getvalue()


# ============================================================
# FLUX画像生成関数
# ============================================================
def generate_castle_image_flux(image_bytes, prompt, key, model_name):
    if not key:
        try:
            key = st.secrets.get("FAL_KEY", "")
        except Exception:
            key = ""

    if not key:
        time.sleep(1.0)
        img = Image.new("RGB", (1024, 1024), color=(150, 160, 170))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        st.warning("⚠️ fal.ai APIキーが入力されていないため、ダミー画像を表示しています。")
        return buf.getvalue()

    st.info(f"🌐 fal.ai / FLUX Kontextで生成中... 使用モデル: {model_name}")

    try:
        os.environ["FAL_KEY"] = key
        import fal_client

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_data_uri = f"data:image/png;base64,{image_base64}"

        full_prompt = prompt + """

Important instruction:
Use the input skeleton image as a structural reference.
Preserve the main silhouette, arch rhythm, tower positions, and overall composition.
Do not reproduce the original black guide lines, dots, baseline, graph-like marks, or wireframe.
Convert the structure into a completed Gaudi-inspired architectural scene.
"""

        result = fal_client.subscribe(
            model_name,
            arguments={
                "prompt": full_prompt,
                "image_url": image_data_uri,
                "guidance_scale": 3.5,
                "num_images": 1,
                "output_format": "png",
                "safety_tolerance": "2",
            },
        )

        if "images" not in result or len(result["images"]) == 0:
            st.error("FLUXから画像URLが返ってきませんでした。")
            with st.expander("fal.aiレスポンス確認"):
                st.json(result)
            return None

        image_url = result["images"][0]["url"]
        img_response = requests.get(image_url, timeout=60)

        if img_response.status_code != 200:
            st.error("生成画像URLから画像を取得できませんでした。")
            st.code(img_response.text)
            return None

        return img_response.content

    except Exception as e:
        st.error(f"fal.ai / FLUX通信中にエラーが発生しました: {e}")
        return None


# ============================================================
# 候補生成関数
# ============================================================
def make_initial_candidate(seed):
    rng = random.Random(seed)
    structure = create_empty_structure()
    add_random_strings(structure, INITIAL_STRINGS, rng)
    simulate_structure(structure)
    structure["parent_string_count"] = 0
    return structure


def make_next_candidate(parent_structure, seed):
    rng = random.Random(seed)

    structure = relax_structure_copy(parent_structure)
    parent_count = len(parent_structure["string_data"])
    structure["parent_string_count"] = parent_count

    added = add_random_strings(structure, ADD_STRINGS_PER_ROUND, rng)
    simulate_structure(structure)

    structure["actually_added"] = added
    return structure


def generate_initial_candidates():
    base_seed = random.randint(0, 10**9)
    st.session_state.candidates = [
        make_initial_candidate(base_seed + i * 1009)
        for i in range(NUM_CHOICES)
    ]


def generate_next_candidates_from_selected():
    parent = st.session_state.selected_structure
    if parent is None:
        return

    base_seed = random.randint(0, 10**9)
    st.session_state.candidates = [
        make_next_candidate(parent, base_seed + i * 1009)
        for i in range(NUM_CHOICES)
    ]


def reset_app():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# ============================================================
# 初期化
# ============================================================
if "app_phase" not in st.session_state:
    st.session_state.app_phase = "choice"
    st.session_state.choice_step = 0
    st.session_state.selected_structure = None
    st.session_state.candidates = []
    st.session_state.generated_image_bytes = None
    st.session_state.ai_input_image_bytes = None
    st.session_state.need_generate_next = False
    generate_initial_candidates()

if (
    st.session_state.app_phase == "choice"
    and st.session_state.choice_step > 0
    and st.session_state.need_generate_next
):
    generate_next_candidates_from_selected()
    st.session_state.need_generate_next = False


# ============================================================
# 画面：4択選択フェーズ
# ============================================================
if st.session_state.app_phase == "choice":
    st.title("🏗️ ガウディ建築の骨組みを選ぶ")

    total_choices = 1 + ADDITION_ROUNDS

    if st.session_state.choice_step == 0:
        st.write("最初に、ランダムに3本のひもを作った候補を4つ表示しています。")
    else:
        st.write("選んだ形をもとに、ランダムに1本のひもを追加した候補を4つ表示しています。")

    st.caption(
        f"選択段階: {st.session_state.choice_step + 1} / {total_choices}　"
        f"最終的なひもの本数: {FINAL_STRING_COUNT}本"
    )

    candidates = st.session_state.candidates

    for row in range(2):
        cols = st.columns(2)
        for col in range(2):
            idx = row * 2 + col
            if idx >= len(candidates):
                continue

            candidate = candidates[idx]

            with cols[col]:
                highlight_new = st.session_state.choice_step > 0

                st.image(
                    draw_structure(candidate, inverted=False, small=True, highlight_new=highlight_new),
                    use_container_width=True
                )

                parent_count = candidate.get("parent_string_count", 0)
                current_count = len(candidate["string_data"])

                if st.session_state.choice_step == 0:
                    st.caption(f"案 {idx + 1}：ひも {current_count}本")
                else:
                    st.caption(
                        f"案 {idx + 1}：{parent_count}本 → {current_count}本 "
                        f"(追加 {candidate.get('actually_added', current_count - parent_count)}本)"
                    )

                if st.button(
                    f"案 {idx + 1} を選択",
                    key=f"select_{st.session_state.choice_step}_{idx}",
                    use_container_width=True
                ):
                    st.session_state.selected_structure = deep_copy_structure(candidate)
                    st.session_state.generated_image_bytes = None
                    st.session_state.ai_input_image_bytes = None

                    if st.session_state.choice_step >= ADDITION_ROUNDS:
                        st.session_state.app_phase = "final"
                    else:
                        st.session_state.choice_step += 1
                        st.session_state.need_generate_next = True

                    st.rerun()

    st.markdown("---")

    if st.button("最初からやり直す", use_container_width=True):
        reset_app()


# ============================================================
# 画面：最終形表示フェーズ
# ============================================================
elif st.session_state.app_phase == "final":
    st.title("✅ 最終的な吊り構造")

    structure = st.session_state.selected_structure

    st.write("選択が完了しました。まずは、吊り下げた状態の最終形を表示しています。")
    st.image(draw_structure(structure, inverted=False, small=False), use_container_width=True)

    st.caption(f"最終的なひもの本数: {len(structure['string_data'])}本")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("上下反転する", type="primary", use_container_width=True):
            st.session_state.app_phase = "inverted"
            st.rerun()

    with col2:
        if st.button("最初からやり直す", use_container_width=True):
            reset_app()


# ============================================================
# 画面：上下反転表示フェーズ
# ============================================================
elif st.session_state.app_phase == "inverted":
    st.title("📐 上下反転した骨組み")

    structure = st.session_state.selected_structure

    st.write("吊り下げた構造を上下反転しました。この画像をFLUX画像生成の入力に使います。")

    ai_input_image_bytes = make_ai_input_image(structure)
    st.session_state.ai_input_image_bytes = ai_input_image_bytes

    st.image(ai_input_image_bytes, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("AI画像を生成する", type="primary", use_container_width=True):
            st.session_state.generated_image_bytes = None
            st.session_state.app_phase = "generate"
            st.rerun()

    with col2:
        if st.button("最終形に戻る", use_container_width=True):
            st.session_state.app_phase = "final"
            st.rerun()


# ============================================================
# 画面：AI生成フェーズ
# ============================================================
elif st.session_state.app_phase == "generate":
    st.title("🏰 FLUX画像生成")

    structure = st.session_state.selected_structure

    if st.session_state.ai_input_image_bytes is None:
        st.session_state.ai_input_image_bytes = make_ai_input_image(structure)

    st.caption(f"現在の生成モデル: {flux_model}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📐 入力骨組み")
        st.image(st.session_state.ai_input_image_bytes, use_container_width=True)

    with col2:
        st.subheader("🎨 FLUX生成結果")

        if st.session_state.generated_image_bytes is None:
            with st.spinner("FLUXがレンダリングしています..."):
                st.session_state.generated_image_bytes = generate_castle_image_flux(
                    st.session_state.ai_input_image_bytes,
                    user_prompt,
                    fal_api_key,
                    flux_model
                )

        if st.session_state.generated_image_bytes:
            st.image(st.session_state.generated_image_bytes, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("別プロンプトで再生成", use_container_width=True):
            st.session_state.generated_image_bytes = None
            st.rerun()

    with col2:
        if st.button("最初からやり直す", use_container_width=True):
            reset_app()
