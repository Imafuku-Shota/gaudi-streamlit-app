import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import time
import requests
import random
import itertools
from PIL import Image
from huggingface_hub import InferenceClient

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
st.sidebar.markdown("APIキーが未入力の場合は、ダミー画像またはエラー表示になります。")

generation_mode = st.sidebar.selectbox(
    "生成方式",
    [
        "Stability AI Control Structure",
        "Hugging Face Inference Providers"
    ]
)

api_key = st.sidebar.text_input(
    "APIキー (Stability AI)",
    type="password",
    help="Stability AIのAPIキーを入力してください"
)

hf_token = st.sidebar.text_input(
    "APIトークン (Hugging Face)",
    type="password",
    help="Hugging Faceの hf_... で始まるアクセストークンを入力してください"
)

hf_provider = st.sidebar.selectbox(
    "Hugging Face Provider",
    [
        "fal-ai",
        "replicate"
    ],
    help="まずは fal-ai で試してください。うまくいかない場合は replicate を試します。"
)

hf_model_id = st.sidebar.selectbox(
    "Hugging Faceモデル",
    [
        "black-forest-labs/FLUX.1-Kontext-dev",
        "black-forest-labs/FLUX.2-dev"
    ],
    help="まずは FLUX.1-Kontext-dev で試してください。"
)

st.sidebar.markdown("---")

user_prompt = st.sidebar.text_area(
    "生成プロンプト",
    value="""Transform this structural skeleton into a completed Gaudi-inspired fantasy castle integrated into a dramatic landscape.

Use the input image only as a loose structural guide for the main silhouette, arches, and tower rhythm. Do not reproduce the guide lines themselves.

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
    """天井固定点だけを持つ空の構造を作る。"""
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
    """候補ごとに独立して扱えるように、構造を深くコピーする。"""
    return {
        "nodes": [dict(n) for n in structure["nodes"]],
        "links": list(structure["links"]),
        "string_data": [dict(s) for s in structure["string_data"]],
        "added_pairs": list(structure.get("added_pairs", []))
    }


def format_node_label(idx):
    """ノード番号を画面表示用の名前に変換する。"""
    if idx < NUM_ANCHORS:
        positions = [
            "一番左", "左から2番目", "左から3番目", "中央の左",
            "中央の右", "右から3番目", "右から2番目", "一番右"
        ]
        return f"天井 {idx + 1} ({positions[idx]})"

    s_id = (idx - NUM_ANCHORS) // NUM_INTERNAL_NODES
    return f"ひも {s_id + 1} の中央"


def add_string_to_structure(structure, idx1, idx2):
    """指定した2点の間に、新しいひもを1本追加する。"""
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
    """
    次にひもを接続できる候補点を返す。
    候補は、8個の天井点と、既存ひもの中央点。
    """
    candidates = list(range(NUM_ANCHORS))

    for s in structure["string_data"]:
        base = NUM_ANCHORS + s["id"] * NUM_INTERNAL_NODES
        middle_node = base + MID_NODE_OFFSET
        if 0 <= middle_node < len(structure["nodes"]):
            candidates.append(middle_node)

    return candidates


def get_existing_pairs(structure):
    """すでに存在するひもの始点・終点ペアを取得する。"""
    existing_pairs = set()
    for s in structure["string_data"]:
        pair = tuple(sorted((s["start_node"], s["end_node"])))
        existing_pairs.add(pair)
    return existing_pairs


def choose_random_pair(structure, rng):
    """現在の構造から、新しく接続する2点をランダムに選ぶ。"""
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
    """ランダムなひもを指定本数だけ追加する。"""
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
    """ひもの物理シミュレーションを行い、垂れ下がった形に落ち着かせる。"""
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
    """既存構造を次候補の親として使う前にコピーする。"""
    copied = deep_copy_structure(structure)
    return copied


def get_active_bounds(structure):
    """描画範囲を現在の構造に合わせて決める。"""
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


def draw_structure(structure, inverted=False, small=False, highlight_new=False, ai_clean=False):
    """
    構造をMatplotlibで描画し、PNGバイト列として返す。

    ai_clean=True の場合：
    AI入力用に、天井点と天井線を消して、骨組みの黒線だけにする。
    """
    fig_size = (3.2, 3.2) if small else (8, 8)
    dpi = 90 if small else 150
    line_width = 2.0 if small else 4.5
    anchor_size = 35 if small else 120

    fig, ax = plt.subplots(figsize=fig_size)

    nodes = structure["nodes"]

    # AI入力用では、黒点と横線を消す
    if not ai_clean:
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
    """AIに渡すため、上下反転した骨組み画像を1024×1024に変換する。"""
    image_bytes = draw_structure(
        structure,
        inverted=True,
        small=False,
        ai_clean=True
    )
    raw_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    resized_img = raw_img.resize((1024, 1024), Image.LANCZOS)

    out_buf = io.BytesIO()
    resized_img.save(out_buf, format="PNG")
    return out_buf.getvalue()


# ============================================================
# AI画像生成関数
# ============================================================
def generate_castle_image_stability(image_bytes, prompt, key):
    """Stability AI Control Structureで、骨組みを構造ガイドとして画像生成する。"""
    if not key:
        time.sleep(1.0)
        img = Image.new("RGB", (1024, 1024), color=(150, 160, 170))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        st.warning("⚠️ Stability AI APIキーが入力されていないため、ダミー画像を表示しています。")
        return buf.getvalue()

    st.info("🌐 Stability AI Control Structureで生成中...")

    url = "https://api.stability.ai/v2beta/stable-image/control/structure"

    try:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {key}",
                "Accept": "image/*"
            },
            files={
                "image": ("skeleton.png", image_bytes, "image/png")
            },
            data={
                "prompt": prompt,
                "control_strength": "0.45",
                "output_format": "png",
                "negative_prompt": (
                    "visible guide lines, black curves, black hanging chains, exposed wireframe, "
                    "scaffolding, skeleton lines, black dots, gray baseline, graph marks, blueprint, "
                    "orthographic front view, flat elevation drawing, centered isolated object, "
                    "toy model, miniature model, plain background, empty background, "
                    "simple line drawing, unfinished sketch, low detail, text, watermark"
                )
            },
            timeout=120
        )

        if response.status_code != 200:
            st.error(f"Stability AI APIエラーが発生しました: {response.status_code}")
            st.code(response.text)
            return None

        return response.content

    except Exception as e:
        st.error(f"Stability AI通信中にエラーが発生しました: {e}")
        return None


def generate_castle_image_huggingface(image_bytes, prompt, token, provider, model_id):
    """Hugging Face Inference Providersのimage-to-imageで画像生成する。"""
    if not token:
        st.warning("⚠️ Hugging Faceトークンが入力されていません。")
        return None

    st.info("🌐 Hugging Face Inference Providersで生成中...")

    hf_prompt = prompt + """

Use the uploaded skeleton image as a loose structural reference.
Preserve the overall rhythm of arches and silhouette.
Do not reproduce black guide lines.
Create a completed architectural scene, not a diagram.
"""

    negative_prompt = (
        "visible guide lines, black curves, black dots, wireframe, blueprint, "
        "flat diagram, simple line drawing, text, watermark, low quality, blurry"
    )

    try:
        client = InferenceClient(
            provider=provider,
            api_key=token
        )

        # Hugging Face公式のimage_to_image形式：
        # input_image bytes + prompt + model
        output_image = client.image_to_image(
            image_bytes,
            prompt=hf_prompt,
            model=model_id,
            negative_prompt=negative_prompt,
            num_inference_steps=28,
            guidance_scale=7.0
        )

        out_buf = io.BytesIO()
        output_image.save(out_buf, format="PNG")
        out_buf.seek(0)
        return out_buf.getvalue()

    except Exception as e:
        st.error(f"Hugging Face通信中にエラーが発生しました: {e}")
        return None


# ============================================================
# 候補生成関数
# ============================================================
def make_initial_candidate(seed):
    """最初の3本線を持つ候補を1つ作る。"""
    rng = random.Random(seed)
    structure = create_empty_structure()
    add_random_strings(structure, INITIAL_STRINGS, rng)
    simulate_structure(structure)
    structure["parent_string_count"] = 0
    return structure


def make_next_candidate(parent_structure, seed):
    """選択済みの構造を親としてコピーし、そこからランダムに1本線を追加した候補を1つ作る。"""
    rng = random.Random(seed)

    structure = relax_structure_copy(parent_structure)
    parent_count = len(parent_structure["string_data"])
    structure["parent_string_count"] = parent_count

    added = add_random_strings(structure, ADD_STRINGS_PER_ROUND, rng)
    simulate_structure(structure)

    structure["actually_added"] = added
    return structure


def generate_initial_candidates():
    """最初の3本線の4候補を生成する。"""
    base_seed = random.randint(0, 10**9)
    st.session_state.candidates = [
        make_initial_candidate(base_seed + i * 1009)
        for i in range(NUM_CHOICES)
    ]


def generate_next_candidates_from_selected():
    """選択済み構造を親として、1本追加した4候補を生成する。"""
    parent = st.session_state.selected_structure
    if parent is None:
        return

    base_seed = random.randint(0, 10**9)
    st.session_state.candidates = [
        make_next_candidate(parent, base_seed + i * 1009)
        for i in range(NUM_CHOICES)
    ]


def reset_app():
    """アプリを最初の状態に戻す。"""
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
    st.write("吊り下げた構造を上下反転しました。この画像をAI画像生成の入力に使います。")

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
    st.title("🏰 AI画像生成")

    structure = st.session_state.selected_structure

    if st.session_state.ai_input_image_bytes is None:
        st.session_state.ai_input_image_bytes = make_ai_input_image(structure)

    st.caption(f"現在の生成方式: {generation_mode}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📐 入力骨組み")
        st.image(st.session_state.ai_input_image_bytes, use_container_width=True)

    with col2:
        st.subheader("🎨 AI生成結果")

        if st.session_state.generated_image_bytes is None:
            with st.spinner("AIがレンダリングしています..."):

                if generation_mode == "Stability AI Control Structure":
                    st.session_state.generated_image_bytes = generate_castle_image_stability(
                        st.session_state.ai_input_image_bytes,
                        user_prompt,
                        api_key
                    )

                elif generation_mode == "Hugging Face Inference Providers":
                    st.session_state.generated_image_bytes = generate_castle_image_huggingface(
                        st.session_state.ai_input_image_bytes,
                        user_prompt,
                        hf_token,
                        hf_provider,
                        hf_model_id
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
