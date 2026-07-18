import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import math
import time
import requests
import random
import itertools
import base64
from PIL import Image

st.set_page_config(layout="wide")

# ============================================================
# 設定値
# ============================================================
# 天井に並べる固定点の数。増やすと接続候補となる天井点が増える。
NUM_ANCHORS = 12

# 天井固定点を配置する横方向の範囲。現在は -17.5 ～ 17.5 に等間隔で配置する。
anchors_x = np.linspace(-17.5, 17.5, NUM_ANCHORS)

# 固定点の座標データ。通常は直接変更せず、NUM_ANCHORSとanchors_xから自動生成する。
anchors = [{"x": float(x), "y": 0.0} for x in anchors_x]

# ひも1本を表現する全ノード数。増やすと曲線が滑らかになるが、計算が重くなる。
NUM_NEW_NODES = 41

# ひもの両端を除いた内部ノード数。NUM_NEW_NODESから自動計算する。
NUM_INTERNAL_NODES = NUM_NEW_NODES - 2

# 内部ノード列の中央位置。通常は直接変更しない。
MID_NODE_OFFSET = NUM_INTERNAL_NODES // 2

# 1回の選択画面に表示する候補数。
NUM_CHOICES = 4

# 最初の候補に作るひもの本数。
INITIAL_STRINGS = 3

# 1回の追加処理で増やすひもの本数。現在の候補生成では直接使われていない。
ADD_STRINGS_PER_ROUND = 1

# 初期選択後に繰り返す追加選択の回数。増やすと最終形までの選択回数が増える。
ADDITION_ROUNDS = 4

# 1ステップごとに下向きへ加える重力。大きくするとひもが強く下へ引かれる。
GRAVITY = 0.08

# ひもが目標の長さを保とうとする強さ。大きいほど硬く、小さいほど伸びやすい。
STIFFNESS = 0.95

# 前ステップの速度を残す割合。小さくすると早く静まり、大きくすると揺れが残りやすい。
DAMPING = 0.75

# 線から線へ接続したとき、横方向の力を接続元へ伝える割合。
# 0.0で完全遮断、1.0で通常の物理挙動。0.1なら横方向の力を10％だけ伝える。
HORIZONTAL_FORCE_SCALE = 0.1

# 縦方向の距離を、ひもの基準長へ追加する割合。
# 大きくすると、高低差がある接続でひもの長さに余裕が増える。
VERTICAL_SAG_RATIO = 0.25

# 横と縦の両方に距離があるとき、追加で長さを与える割合。
# min(dx, dy)に掛けるため、斜め方向の接続にだけ強く効く。
DIAGONAL_SAG_RATIO = 0.15

# 基準長を直線距離の何倍まで許可するか。
# 追加分が大きくなっても、垂れすぎないように制限する。
MAX_BASE_DISTANCE_RATIO = 1.25

# ひもの長さ倍率の下限。大きくすると、最も短いひもでもたるみやすくなる。
STRING_LENGTH_SCALE_MIN = 1.01

# ひもの長さ倍率の上限。大きくすると、深く垂れる長いひもが作られやすくなる。
STRING_LENGTH_SCALE_MAX = 1.06

# 接続を許可する2点間の最小水平距離。大きくすると短い横幅のひもが作られにくくなる。
MIN_HORIZONTAL_DISTANCE = 3.0

# ============================================================
# Secrets 読み込み
# ============================================================
def read_secret(name):
    """Streamlit Secretsから値を読む。未設定なら None を返す。"""
    try:
        value = st.secrets.get(name, None)
    except Exception:
        value = None

    if value == "":
        return None

    return value


stability_api_key = read_secret("STABILITY_API_KEY")
openai_api_key = read_secret("OPENAI_API_KEY")

# ============================================================
# サイドバー：AI生成設定
# ============================================================
st.sidebar.title("⚙️ AI生成設定")

api_provider = st.sidebar.radio(
    "画像生成API",
    ["Stability AI", "OpenAI"],
    index=0,
    help="画像生成に使うAPIを選択します。"
)

st.sidebar.markdown("#### APIキー状態")

if stability_api_key:
    st.sidebar.success("Stability AI：設定済み")
else:
    st.sidebar.warning("Stability AI：未設定")

if openai_api_key:
    st.sidebar.success("OpenAI：設定済み")
else:
    st.sidebar.warning("OpenAI：未設定")

if api_provider == "Stability AI":
    selected_api_key = stability_api_key
    if not selected_api_key:
        st.sidebar.warning("現在 Stability AI が選択されていますが、STABILITY_API_KEY が未設定です。")
else:
    selected_api_key = openai_api_key
    if not selected_api_key:
        st.sidebar.warning("現在 OpenAI が選択されていますが、OPENAI_API_KEY が未設定です。")

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
    copied = {
        "nodes": [dict(n) for n in structure["nodes"]],
        "links": list(structure["links"]),
        "string_data": [dict(s) for s in structure["string_data"]],
        "added_pairs": list(structure.get("added_pairs", []))
    }

    optional_keys = [
        "parent_string_count",
        "action_performed",
        "actions_taken",
        "actually_added",
        "highlighted_string_id",
        "highlighted_string_ids",
        "deleted_string_path",
        "deleted_string_paths",
        "change_signature"
    ]

    for key in optional_keys:
        if key in structure:
            if key == "deleted_string_path":
                copied[key] = [tuple(p) for p in structure[key]]
            elif key == "deleted_string_paths":
                copied[key] = [
                    [tuple(point) for point in path]
                    for path in structure[key]
                ]
            elif key in {"actions_taken", "highlighted_string_ids"}:
                copied[key] = list(structure[key])
            else:
                copied[key] = structure[key]

    return copied


def clear_change_display_metadata(structure):
    """
    前の選択段階で使用した青表示情報だけを消す。
    構造や物理状態には手を加えない。
    """
    for key in [
        "highlighted_string_id",
        "highlighted_string_ids",
        "deleted_string_path",
        "deleted_string_paths"
    ]:
        structure.pop(key, None)

    return structure


def format_node_label(idx):
    """ノード番号を画面表示用の名前に変換する。"""
    if idx < NUM_ANCHORS:
        return f"天井 {idx + 1}"

    s_id = (idx - NUM_ANCHORS) // NUM_INTERNAL_NODES
    return f"ひも {s_id + 1} の一番下"


def get_active_strings(structure):
    """削除されていない有効なひもだけを返す。"""
    return [
        s for s in structure["string_data"]
        if not s.get("is_deleted", False)
    ]


def count_active_strings(structure):
    """削除されていない有効なひもの本数を数える。"""
    return len(get_active_strings(structure))


def get_string_internal_start(s):
    """指定したひもの内部ノード列の開始番号を返す。"""
    return s.get(
        "internal_node_start",
        NUM_ANCHORS + s["id"] * NUM_INTERNAL_NODES
    )


def get_string_logical_nodes(s):
    """仮想固定点ではなく、本来の接続先ノード番号を返す。"""
    return (
        s.get("logical_start_node", s["start_node"]),
        s.get("logical_end_node", s["end_node"])
    )


def create_vertical_follow_anchor(structure, source_node_idx):
    """
    指定したノードのX座標を追従し、Y方向の力だけを
    接続元へ伝える仮想接続点を作る。

    新しいひもからの横方向の力は接続元へ伝えない。
    """
    source = structure["nodes"][source_node_idx]

    structure["nodes"].append({
        "x": float(source["x"]),
        "y": float(source["y"]),
        "px": float(source["x"]),
        "py": float(source["y"]),
        "fixed": True,
        "vertical_follow_node": source_node_idx
    })

    return len(structure["nodes"]) - 1


def prepare_attachment_node(structure, node_idx):
    """
    天井固定点はそのまま使う。
    既存のひもの一番下の点へ接続する場合は、
    横方向の力を遮断し、縦方向の力だけを伝える仮想点を作る。
    """
    if node_idx < NUM_ANCHORS:
        return node_idx

    return create_vertical_follow_anchor(structure, node_idx)


def get_string_bottom_node(structure, s):
    """指定したひもの内部ノードのうち、一番下にあるノード番号を返す。"""
    base = get_string_internal_start(s)
    valid_indices = [
        idx
        for idx in range(base, base + NUM_INTERNAL_NODES)
        if 0 <= idx < len(structure["nodes"])
    ]

    if not valid_indices:
        return base + MID_NODE_OFFSET

    return min(
        valid_indices,
        key=lambda idx: structure["nodes"][idx]["y"]
    )


def get_string_node_indices(s):
    """指定したひもを構成するノード番号の一覧を返す。"""
    base = get_string_internal_start(s)
    return (
        [s["start_node"]]
        + list(range(base, base + NUM_INTERNAL_NODES))
        + [s["end_node"]]
    )

def capture_string_path(structure, s):
    """削除前のひもの形を保存するため、座標列を取得する。"""
    nodes = structure["nodes"]
    node_indices = get_string_node_indices(s)

    path = []
    for i in node_indices:
        if 0 <= i < len(nodes):
            path.append((float(nodes[i]["x"]), float(nodes[i]["y"])))

    return path


def get_string_bottom_y(structure, s):
    """
    指定したひもの一番下のy座標を返す。
    このコードでは、下に垂れるほど y が小さくなるので min を使う。
    """
    nodes = structure["nodes"]
    node_indices = get_string_node_indices(s)

    ys = [
        nodes[i]["y"]
        for i in node_indices
        if 0 <= i < len(nodes)
    ]

    if not ys:
        return 0.0

    return min(ys)


def string_has_child(target_s, active_strings):
    """
    target_s の中央点に、他の有効なひもが論理的に接続しているか確認する。
    """
    middle_node = get_string_middle_node(target_s)

    for other in active_strings:
        if other["id"] == target_s["id"]:
            continue

        other_start, other_end = get_string_logical_nodes(other)
        if other_start == middle_node or other_end == middle_node:
            return True

    return False

def get_leaf_strings(structure):
    """
    何もぶら下がっていない末端のひもだけを返す。
    つまり、そのひもの中央点に他のひもが接続していないものだけを返す。
    """
    active_strings = get_active_strings(structure)
    leaf_strings = []

    for s in active_strings:
        if not string_has_child(s, active_strings):
            leaf_strings.append(s)

    return leaf_strings


def get_bottom_leaf_strings(structure):
    """
    削除可能な末端のひもの中で、一番下にあるひもだけを返す。
    一番下の高さが同じものが複数ある場合は、複数返す。
    """
    leaf_strings = get_leaf_strings(structure)

    if not leaf_strings:
        return []

    bottom_y = min(get_string_bottom_y(structure, s) for s in leaf_strings)

    bottom_leaf_strings = [
        s for s in leaf_strings
        if abs(get_string_bottom_y(structure, s) - bottom_y) < 1e-6
    ]

    return bottom_leaf_strings


def get_string_by_id(structure, string_id):
    """string_data から指定IDのひもを探す。"""
    for s in structure["string_data"]:
        if s["id"] == string_id:
            return s
    return None


def add_string_to_structure(structure, idx1, idx2, rng=None):
    """
    指定した2点の間に、新しいひもを1本追加する。

    既存のひもの一番下へ接続するときは仮想接続点を使用し、
    横方向の力を遮断して、縦方向の力だけを元のひもへ伝える。
    """
    nodes = structure["nodes"]
    links = structure["links"]
    string_data = structure["string_data"]

    logical_idx1 = idx1
    logical_idx2 = idx2

    physics_idx1 = prepare_attachment_node(structure, logical_idx1)
    physics_idx2 = prepare_attachment_node(structure, logical_idx2)

    p1 = nodes[physics_idx1]
    p2 = nodes[physics_idx2]

    dx = abs(p2["x"] - p1["x"])
    dy = abs(p2["y"] - p1["y"])

    # 従来の2点間の直線距離
    straight_dist = math.hypot(dx, dy)

    if straight_dist == 0:
        return False

    # 高低差がある接続には、縦方向の距離に応じて長さを追加する
    vertical_extra = dy * VERTICAL_SAG_RATIO

    # 横と縦の両方に距離がある斜め接続には、さらに長さを追加する
    diagonal_extra = min(dx, dy) * DIAGONAL_SAG_RATIO

    base_dist = straight_dist + vertical_extra + diagonal_extra

    # 長くなりすぎて深く垂れないよう、直線距離に対する上限を設ける
    max_base_dist = straight_dist * MAX_BASE_DISTANCE_RATIO
    dist = min(base_dist, max_base_dist)

    # rngが渡されていない場合も動作できるようにする
    if rng is None:
        rng = random

    length_scale = rng.uniform(
        STRING_LENGTH_SCALE_MIN,
        STRING_LENGTH_SCALE_MAX
    )
    total_len = dist * length_scale
    seg_len = total_len / (NUM_NEW_NODES - 1)

    new_x = np.linspace(p1["x"], p2["x"], NUM_NEW_NODES)
    new_y = np.zeros(NUM_NEW_NODES)

    sag_depth = dist * 0.01
    for i in range(NUM_NEW_NODES):
        t = i / (NUM_NEW_NODES - 1)
        linear_y = p1["y"] + (p2["y"] - p1["y"]) * t
        sag = sag_depth * (4.0 * t * (1.0 - t))
        new_y[i] = linear_y - sag

    internal_node_start = len(nodes)
    first_link_idx = len(links)

    for i in range(1, NUM_NEW_NODES - 1):
        nodes.append({
            "x": float(new_x[i]),
            "y": float(new_y[i]),
            "px": float(new_x[i]),
            "py": float(new_y[i]),
            "fixed": False
        })

    prev_idx = physics_idx1
    current_new_idx = internal_node_start

    for _ in range(1, NUM_NEW_NODES - 1):
        links.append((prev_idx, current_new_idx, seg_len))
        prev_idx = current_new_idx
        current_new_idx += 1

    links.append((prev_idx, physics_idx2, seg_len))

    string_data.append({
        "id": len(string_data),
        "start_node": physics_idx1,
        "end_node": physics_idx2,
        "logical_start_node": logical_idx1,
        "logical_end_node": logical_idx2,
        "internal_node_start": internal_node_start,
        "first_link_idx": first_link_idx,
        "last_link_idx": len(links) - 1,
        "is_deleted": False
    })

    structure.setdefault("added_pairs", []).append((logical_idx1, logical_idx2))

    return True

def get_connection_candidates(structure):
    """天井点と、各有効ひもの一番下のノードを接続候補として返す。"""
    candidates = list(range(NUM_ANCHORS))

    for s in structure["string_data"]:
        if s.get("is_deleted", False):
            continue

        bottom_node = get_string_bottom_node(structure, s)

        if 0 <= bottom_node < len(structure["nodes"]):
            candidates.append(bottom_node)

    return candidates


def get_existing_pairs(structure):
    """現在存在するひもの論理的な接続ペアを取得する。"""
    existing_pairs = set()

    for string_info in structure["string_data"]:
        if string_info.get("is_deleted", False):
            continue

        start_node, end_node = get_string_logical_nodes(string_info)
        existing_pairs.add(tuple(sorted((start_node, end_node))))

    return existing_pairs

def get_valid_add_pairs(structure):
    """追加可能な接続ペアをすべて返す。"""
    candidates = get_connection_candidates(structure)

    if len(candidates) < 2:
        return []

    existing_pairs = get_existing_pairs(structure)
    all_pairs = list(itertools.combinations(candidates, 2))

    valid_pairs = []

    for p in all_pairs:
        sorted_pair = tuple(sorted(p))

        if sorted_pair in existing_pairs:
            continue

        n1 = structure["nodes"][p[0]]
        n2 = structure["nodes"][p[1]]
        dx = abs(n2["x"] - n1["x"])

        if dx < MIN_HORIZONTAL_DISTANCE:
            continue

        valid_pairs.append(sorted_pair)

    return sorted(set(valid_pairs))


def choose_random_pair(structure, rng):
    """現在の構造から、新しく接続する2点をランダムに選ぶ。"""
    valid_pairs = get_valid_add_pairs(structure)

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

        ok = add_string_to_structure(structure, pair[0], pair[1], rng=rng)
        if ok:
            added += 1

    return added


def get_valid_reconnect_changes(structure):
    """付け直しとして実行可能な変化をすべて返す。"""
    active_strings = get_active_strings(structure)

    if not active_strings:
        return []

    candidates = get_connection_candidates(structure)
    existing_pairs = get_existing_pairs(structure)
    changes = []

    for string_info in active_strings:
        current_start, current_end = get_string_logical_nodes(string_info)
        current_pair = tuple(sorted((current_start, current_end)))
        my_bottom = get_string_bottom_node(structure, string_info)

        existing_pairs_without_self = set(existing_pairs)
        existing_pairs_without_self.discard(current_pair)

        for end_type in ["start", "end"]:
            fixed_other = current_end if end_type == "start" else current_start

            for new_target in candidates:
                if new_target in {current_start, current_end, my_bottom, fixed_other}:
                    continue

                new_pair = tuple(sorted((new_target, fixed_other)))
                if new_pair in existing_pairs_without_self:
                    continue

                n1 = structure["nodes"][new_target]
                n2 = structure["nodes"][fixed_other]
                dx = abs(n2["x"] - n1["x"])

                if dx < MIN_HORIZONTAL_DISTANCE:
                    continue

                changes.append(("reconnect", string_info["id"], end_type, new_target))

    return sorted(set(changes))


def get_possible_changes_by_action(structure):
    """現在の構造から、重複しない実行可能な変化を行動別に返す。"""
    add_changes = [
        ("add", pair[0], pair[1])
        for pair in get_valid_add_pairs(structure)
    ]

    delete_changes = [
        ("delete", s["id"])
        for s in get_bottom_leaf_strings(structure)
    ]

    reconnect_changes = get_valid_reconnect_changes(structure)

    return {
        "add": add_changes,
        "delete": delete_changes,
        "reconnect": reconnect_changes
    }


def change_signature(change):
    """変化の重複判定用の署名を返す。"""
    return tuple(change)


def structure_signature(structure):
    """
    構造の重複判定用の署名を返す。
    仮想固定点ではなく、本来の接続先で判定する。
    """
    items = []

    for string_info in get_active_strings(structure):
        start_node, end_node = get_string_logical_nodes(string_info)
        pair = tuple(sorted((start_node, end_node)))
        items.append((string_info["id"], pair[0], pair[1]))

    return tuple(sorted(items))

def simulate_structure(structure, steps=350, constraint_iterations=6):
    """
    ひもの物理シミュレーションを行い、垂れ下がった形に落ち着かせる。

    vertical_follow_nodeを持つ仮想接続点は、接続元のX座標だけを追従する。
    新しいひもから受ける横方向の補正は設定した割合だけ伝え、
    縦方向の補正はそのまま接続元ノードへ伝える。
    """
    nodes = structure["nodes"]

    active_links = [
        (idx1, idx2, target_dist)
        for idx1, idx2, target_dist in structure["links"]
        if not (idx1 == 0 and idx2 == 0)
    ]

    movable_nodes = [
        node
        for node in nodes
        if not node.get("fixed", False)
    ]

    vertical_follow_nodes = [
        node
        for node in nodes
        if "vertical_follow_node" in node
    ]

    def sync_vertical_follow_node(node):
        source_idx = node["vertical_follow_node"]
        if not (0 <= source_idx < len(nodes)):
            return None

        source = nodes[source_idx]
        node["x"] = source["x"]
        node["y"] = source["y"]
        node["px"] = source["x"]
        node["py"] = source["y"]
        return source

    def apply_node_correction(node, correction_x, correction_y):
        """通常点にはXY補正、仮想接続点には横を減衰させたXY補正を適用する。"""
        if "vertical_follow_node" in node:
            source = sync_vertical_follow_node(node)
            if source is None:
                return

            # 横方向は設定した割合だけ、縦方向はそのまま接続元へ伝える
            source["x"] += correction_x * HORIZONTAL_FORCE_SCALE
            source["y"] += correction_y
            node["x"] = source["x"]
            node["y"] = source["y"]
            node["px"] = source["x"]
            node["py"] = source["y"]
            return

        if not node.get("fixed", False):
            node["x"] += correction_x
            node["y"] += correction_y

    for _ in range(steps):
        for node in movable_nodes:
            vx = (node["x"] - node["px"]) * DAMPING
            vy = (node["y"] - node["py"]) * DAMPING

            node["px"] = node["x"]
            node["py"] = node["y"]

            node["x"] += vx
            node["y"] += vy - GRAVITY

        for _ in range(constraint_iterations):
            for node in vertical_follow_nodes:
                sync_vertical_follow_node(node)

            for idx1, idx2, target_dist in active_links:
                n1 = nodes[idx1]
                n2 = nodes[idx2]

                if "vertical_follow_node" in n1:
                    sync_vertical_follow_node(n1)
                if "vertical_follow_node" in n2:
                    sync_vertical_follow_node(n2)

                dx = n2["x"] - n1["x"]
                dy = n2["y"] - n1["y"]
                distance = math.hypot(dx, dy)

                if distance == 0:
                    continue

                diff = (target_dist - distance) / distance * 0.5
                correction_x = dx * diff * STIFFNESS
                correction_y = dy * diff * STIFFNESS

                apply_node_correction(n1, -correction_x, -correction_y)
                apply_node_correction(n2, correction_x, correction_y)


def relax_structure_copy(structure):
    """選択済み構造を直接変更しないためにコピーする。"""
    copied = deep_copy_structure(structure)
    return copied


def get_active_bounds(structure):
    """描画範囲を現在の構造に合わせて決める。"""
    nodes = structure["nodes"]
    active_node_indices = set(range(NUM_ANCHORS))

    for string_info in structure["string_data"]:
        if string_info.get("is_deleted", False):
            continue

        base = get_string_internal_start(string_info)
        active_node_indices.update(range(base, base + NUM_INTERNAL_NODES))

    ys = [
        nodes[i]["y"]
        for i in active_node_indices
        if isinstance(i, int) and 0 <= i < len(nodes)
    ]

    deleted_paths = list(structure.get("deleted_string_paths", []))

    # 旧形式のデータにも対応する
    deleted_path = structure.get("deleted_string_path", None)
    if deleted_path:
        deleted_paths.append(deleted_path)

    for path in deleted_paths:
        ys.extend([point[1] for point in path])

    min_y = min(ys) if ys else -15.0
    bottom_limit = min_y - 5.0
    half_width = max(20.0, abs(bottom_limit) * 0.8)

    return bottom_limit, half_width

def draw_structure(structure, inverted=False, small=False, highlight_new=False):
    """構造をMatplotlibで描画し、PNGバイト列として返す。"""
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
        if s.get("is_deleted", False):
            continue

        node_indices = get_string_node_indices(s)

        xs = [nodes[i]["x"] for i in node_indices if 0 <= i < len(nodes)]
        ys = [nodes[i]["y"] for i in node_indices if 0 <= i < len(nodes)]

        is_highlighted = False
        if highlight_new:
            highlighted_ids = set(structure.get("highlighted_string_ids", []))

            # 旧形式のデータにも対応する
            if "highlighted_string_id" in structure:
                highlighted_ids.add(structure["highlighted_string_id"])

            if highlighted_ids:
                is_highlighted = s["id"] in highlighted_ids
            else:
                is_highlighted = (
                    parent_string_count is not None
                    and s["id"] >= parent_string_count
                )

        color = "#1C83E1" if is_highlighted else "black"
        lw = line_width + 1.0 if is_highlighted else line_width
        z_order = 8 if is_highlighted else 6

        ax.plot(
            xs,
            ys,
            color=color,
            lw=lw,
            solid_joinstyle="round",
            solid_capstyle="round",
            zorder=z_order
        )

    # 削除したひもは、削除前の形を青の点線で重ねて表示する
    if highlight_new:
        deleted_paths = list(structure.get("deleted_string_paths", []))

        # 旧形式のデータにも対応する
        deleted_path = structure.get("deleted_string_path", None)
        if deleted_path:
            deleted_paths.append(deleted_path)

        for path in deleted_paths:
            if not path:
                continue

            deleted_xs = [point[0] for point in path]
            deleted_ys = [point[1] for point in path]

            ax.plot(
                deleted_xs,
                deleted_ys,
                color="#1C83E1",
                lw=line_width + 1.0,
                linestyle=(0, (1.0, 2.0)),
                dash_capstyle="round",
                solid_joinstyle="round",
                zorder=9
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
    image_bytes = draw_structure(structure, inverted=True, small=False)
    raw_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    resized_img = raw_img.resize((1024, 1024), Image.LANCZOS)

    out_buf = io.BytesIO()
    resized_img.save(out_buf, format="PNG")
    return out_buf.getvalue()


# ============================================================
# 画像生成API
# ============================================================
def make_dummy_image():
    """APIキー未設定時に表示するダミー画像を返す。"""
    time.sleep(1.0)
    img = Image.new("RGB", (1024, 1024), color=(150, 160, 170))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate_stability_image(image_bytes, prompt, key):
    """Stability AI Control Structureで、骨組みを構造ガイドとして画像生成する。"""
    if not key:
        st.warning("⚠️ STABILITY_API_KEY が設定されていないため、ダミー画像を表示しています。")
        return make_dummy_image()

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
                    "black guide lines, dots, graph marks, wireframe, blueprint, "
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
        st.error(f"Stability AI 通信中にエラーが発生しました: {e}")
        return None


def generate_openai_image(image_bytes, prompt, key):
    """OpenAI Images Edit APIで、骨組み画像を入力として画像生成する。"""
    if not key:
        st.warning("⚠️ OPENAI_API_KEY が設定されていないため、ダミー画像を表示しています。")
        return make_dummy_image()

    st.info("🌐 OpenAI Images APIで生成中...")

    url = "https://api.openai.com/v1/images/edits"

    try:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {key}"
            },
            files=[
                (
                    "image[]",
                    ("skeleton.png", image_bytes, "image/png")
                )
            ],
            data={
                "model": "gpt-image-1.5",
                "prompt": prompt,
                "size": "1024x1024",
                "quality": "medium",
                "output_format": "png",
                "n": "1"
            },
            timeout=180
        )

        if response.status_code != 200:
            st.error(f"OpenAI APIエラーが発生しました: {response.status_code}")
            st.code(response.text)
            return None

        data = response.json()

        if "data" not in data or len(data["data"]) == 0:
            st.error("OpenAI APIの返答に画像データが含まれていません。")
            st.code(data)
            return None

        b64_image = data["data"][0].get("b64_json")
        if not b64_image:
            st.error("OpenAI APIの返答に b64_json が含まれていません。")
            st.code(data)
            return None

        return base64.b64decode(b64_image)

    except Exception as e:
        st.error(f"OpenAI 通信中にエラーが発生しました: {e}")
        return None


def generate_castle_image(image_bytes, prompt, provider, key):
    """選択されたAPIで画像生成する。"""
    if provider == "Stability AI":
        return generate_stability_image(image_bytes, prompt, key)

    if provider == "OpenAI":
        return generate_openai_image(image_bytes, prompt, key)

    st.error("画像生成APIの選択が不正です。")
    return None


def make_initial_candidate(seed):
    """最初の5本線を持つ候補を1つ作る。"""
    rng = random.Random(seed)
    structure = create_empty_structure()
    add_random_strings(structure, INITIAL_STRINGS, rng)
    simulate_structure(structure)
    structure["parent_string_count"] = 0
    structure["action_performed"] = "initial"
    structure["actually_added"] = INITIAL_STRINGS
    structure["change_signature"] = ("initial", structure_signature(structure))
    return structure


# ============================================================
# 変化を適用する処理
# ============================================================
def apply_add_change(parent_structure, change, rng=None):
    """指定された追加変化を適用した候補を作る。"""
    structure = relax_structure_copy(parent_structure)

    parent_count = count_active_strings(parent_structure)
    structure["parent_string_count"] = parent_count

    _, idx1, idx2 = change

    ok = add_string_to_structure(structure, idx1, idx2, rng=rng)
    simulate_structure(structure)

    structure["action_performed"] = "add"
    structure["actually_added"] = 1 if ok else 0
    structure["change_signature"] = change_signature(change)

    if ok:
        new_string_id = structure["string_data"][-1]["id"]
        highlighted_ids = structure.setdefault("highlighted_string_ids", [])
        if new_string_id not in highlighted_ids:
            highlighted_ids.append(new_string_id)

    return structure


def apply_delete_change(parent_structure, change):
    """
    指定された削除変化を適用した候補を作る。
    削除前のひもの座標を保存して、表示時に青の点線で描く。
    """
    structure = relax_structure_copy(parent_structure)

    parent_count = count_active_strings(parent_structure)
    structure["parent_string_count"] = parent_count

    _, target_id = change
    target_s = get_string_by_id(structure, target_id)

    if target_s is None or target_s.get("is_deleted", False):
        structure["action_performed"] = "delete"
        structure["actually_added"] = 0
        structure["change_signature"] = change_signature(change)
        return structure

    # 削除前の形を保存し、同じ候補内の複数変更をすべて表示できるようにする
    deleted_path = capture_string_path(structure, target_s)
    structure.setdefault("deleted_string_paths", []).append(deleted_path)

    structure["action_performed"] = "delete"
    structure["actually_added"] = 0
    structure["change_signature"] = change_signature(change)

    target_s["is_deleted"] = True

    for i in range(target_s["first_link_idx"], target_s["last_link_idx"] + 1):
        structure["links"][i] = (0, 0, 0)

    base = get_string_internal_start(target_s)
    for i in range(base, base + NUM_INTERNAL_NODES):
        if 0 <= i < len(structure["nodes"]):
            structure["nodes"][i]["fixed"] = True
            structure["nodes"][i]["x"] = 0.0
            structure["nodes"][i]["y"] = 9999.0

    simulate_structure(structure)

    return structure


def apply_reconnect_change(parent_structure, change):
    """指定された付け直し変化を適用した候補を作る。"""
    structure = relax_structure_copy(parent_structure)

    parent_count = count_active_strings(parent_structure)
    structure["parent_string_count"] = parent_count

    _, target_id, end_type, new_target = change
    target_s = get_string_by_id(structure, target_id)

    if target_s is None or target_s.get("is_deleted", False):
        structure["action_performed"] = "reconnect"
        structure["actually_added"] = 0
        structure["change_signature"] = change_signature(change)
        return structure

    # つなぎ直す前の形を保存する。
    # つなぎ直し後に消えた旧位置を、青の点線で表示するために使う。
    old_reconnect_path = capture_string_path(structure, target_s)
    structure.setdefault("deleted_string_paths", []).append(old_reconnect_path)

    physics_target = prepare_attachment_node(structure, new_target)

    if end_type == "start":
        link_idx = target_s["first_link_idx"]
        old_link = structure["links"][link_idx]
        structure["links"][link_idx] = (physics_target, old_link[1], old_link[2])
        target_s["start_node"] = physics_target
        target_s["logical_start_node"] = new_target

    elif end_type == "end":
        link_idx = target_s["last_link_idx"]
        old_link = structure["links"][link_idx]
        structure["links"][link_idx] = (old_link[0], physics_target, old_link[2])
        target_s["end_node"] = physics_target
        target_s["logical_end_node"] = new_target

    structure["action_performed"] = "reconnect"
    structure["actually_added"] = 0

    highlighted_ids = structure.setdefault("highlighted_string_ids", [])
    if target_s["id"] not in highlighted_ids:
        highlighted_ids.append(target_s["id"])

    structure["change_signature"] = change_signature(change)

    simulate_structure(structure)

    return structure

def apply_change(parent_structure, change, rng=None):
    """変化の種類に応じて候補を作る。"""
    action = change[0]

    if action == "add":
        return apply_add_change(parent_structure, change, rng=rng)

    if action == "delete":
        return apply_delete_change(parent_structure, change)

    if action == "reconnect":
        return apply_reconnect_change(parent_structure, change)

    return relax_structure_copy(parent_structure)


def generate_unique_next_candidates(parent_structure, num_choices):
    """
    選択済み構造を親として、
    1. 既存のひもを1本つなぎ直す
    2. 新しいひもを1本追加する
    の順で候補を生成する。
    """
    rng = random.Random(random.randint(0, 10**9))

    candidates = []
    used_structure_signatures = set()
    parent_signature = structure_signature(parent_structure)

    # 親構造に対するつなぎ直し候補は最初に1回だけ計算する
    reconnect_changes = get_valid_reconnect_changes(parent_structure)

    if not reconnect_changes:
        return []

    attempts = 0
    max_attempts = 1000

    while len(candidates) < num_choices and attempts < max_attempts:
        attempts += 1

        # 前段階の青表示を引き継がず、今回の2変化だけを記録する
        candidate_base = clear_change_display_metadata(
            deep_copy_structure(parent_structure)
        )

        reconnect_change = rng.choice(reconnect_changes)
        candidate = apply_change(candidate_base, reconnect_change)

        add_changes = get_valid_add_pairs(candidate)

        if not add_changes:
            continue

        idx1, idx2 = rng.choice(add_changes)
        add_change = ("add", idx1, idx2)

        candidate = apply_change(candidate, add_change, rng=rng)
        candidate_signature = structure_signature(candidate)

        if candidate_signature == parent_signature:
            continue

        if candidate_signature in used_structure_signatures:
            continue

        candidate["action_performed"] = "reconnect_then_add"
        candidate["actions_taken"] = ["reconnect", "add"]

        used_structure_signatures.add(candidate_signature)
        candidates.append(candidate)

    return candidates


def generate_initial_candidates():
    """最初の5本線の4候補を生成する。"""
    candidates = []
    used_signatures = set()

    attempts = 0
    max_attempts = 200

    while len(candidates) < NUM_CHOICES and attempts < max_attempts:
        attempts += 1
        seed = random.randint(0, 10**9)
        candidate = make_initial_candidate(seed)
        sig = structure_signature(candidate)

        if sig in used_signatures:
            continue

        used_signatures.add(sig)
        candidates.append(candidate)

    st.session_state.candidates = candidates


def generate_next_candidates_from_selected():
    """選択済み構造を親として、重複しないランダムな変化を加えた候補を生成する。"""
    parent = st.session_state.selected_structure
    if parent is None:
        return

    st.session_state.candidates = generate_unique_next_candidates(
        parent,
        NUM_CHOICES
    )


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
    candidates = st.session_state.candidates

    if len(candidates) < NUM_CHOICES:
        st.warning(
            f"重複しない変化だけを表示しているため、候補数が {len(candidates)} 個になっています。"
        )

    def render_candidate_grid(candidate_list):
        """候補を2列×2行で表示する。"""
        for row in range(2):
            cols = st.columns(2)

            for col in range(2):
                idx = row * 2 + col

                if idx >= len(candidate_list):
                    continue

                candidate = candidate_list[idx]

                with cols[col]:
                    # 候補カードをカラム幅の中央50％に配置し、
                    # ワイド画面でも画像が大きくなりすぎないようにする。
                    candidate_inner_cols = st.columns([0.25, 0.50, 0.25])

                    with candidate_inner_cols[1]:
                        highlight_new = st.session_state.choice_step > 0

                        st.image(
                            draw_structure(
                                candidate,
                                inverted=False,
                                small=True,
                                highlight_new=highlight_new
                            ),
                            use_container_width=True
                        )

                        current_count = count_active_strings(candidate)

                        if st.session_state.choice_step == 0:
                            st.caption(
                                f"案 {idx + 1}：初期ひも {current_count}本"
                            )
                        else:
                            act = candidate.get("action_performed", "add")

                            if act == "reconnect_then_add":
                                st.caption(
                                    f"案 {idx + 1}：🔄 つなぎ直し → "
                                    f"🟢 ひも追加（計 {current_count}本）"
                                )
                            else:
                                action_labels = {
                                    "add": "🟢 ひも追加",
                                    "delete": "🗑️ ひも削除（削除前を青点線）",
                                    "reconnect": "🔄 つなぎ直し",
                                    "initial": "初期ひも"
                                }

                                label = action_labels.get(act, "ひも追加")
                                st.caption(
                                    f"案 {idx + 1}：{label}（計 {current_count}本）"
                                )

                        if st.button(
                            f"案 {idx + 1} を選択",
                            key=f"select_{st.session_state.choice_step}_{idx}",
                            use_container_width=True
                        ):
                            st.session_state.selected_structure = deep_copy_structure(
                                candidate
                            )
                            st.session_state.generated_image_bytes = None
                            st.session_state.ai_input_image_bytes = None

                            if st.session_state.choice_step >= ADDITION_ROUNDS:
                                st.session_state.app_phase = "final"
                            else:
                                st.session_state.choice_step += 1
                                st.session_state.need_generate_next = True

                            st.rerun()

    if (
        st.session_state.choice_step > 0
        and st.session_state.selected_structure is not None
    ):
        # 2回目以降は、左に選択済み骨格、右に候補を配置する
        left_col, right_col = st.columns(
            [1.0, 2.8],
            gap="large"
        )

        with left_col:
            st.subheader("前回選択した骨組み")

            selected_count = count_active_strings(
                st.session_state.selected_structure
            )

            selected_inner_cols = st.columns([0.08, 0.84, 0.08])

            with selected_inner_cols[1]:
                st.image(
                    draw_structure(
                        st.session_state.selected_structure,
                        inverted=False,
                        small=True,
                        highlight_new=False
                    ),
                    use_container_width=True
                )

                st.caption(
                    f"選択中の骨組み：ひも {selected_count}本"
                )

            st.caption(
                f"選択段階：{st.session_state.choice_step + 1} / "
                f"{total_choices}"
            )

        with right_col:
            st.subheader("次の候補")

            st.write(
                "選んだ形をもとに、既存のひもを1本つなぎ直した後、"
                "新しいひもを1本追加した候補を表示しています。"
            )

            render_candidate_grid(candidates)

    else:
        # 初回は選択済み骨格がないため、候補を画面全体に表示する
        st.write(
            f"最初に、ランダムに{INITIAL_STRINGS}本のひもを作った"
            "候補を4つ表示しています。"
        )

        st.caption(
            f"選択段階：{st.session_state.choice_step + 1} / "
            f"{total_choices}"
        )

        render_candidate_grid(candidates)

    st.markdown("---")

    if st.button(
        "最初からやり直す",
        use_container_width=True
    ):
        reset_app()


# ============================================================
# 画面：最終形表示フェーズ
# ============================================================
elif st.session_state.app_phase == "final":
    st.title("✅ 最終的な吊り構造")

    structure = st.session_state.selected_structure
    st.write("選択が完了しました。まずは、吊り下げた状態の最終形を表示しています。")
    st.image(draw_structure(structure, inverted=False, small=False), use_container_width=True)

    final_count = count_active_strings(structure)
    st.caption(f"最終的なひもの本数: {final_count}本")

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

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📐 入力骨組み")
        st.image(st.session_state.ai_input_image_bytes, use_container_width=True)

    with col2:
        st.subheader("🎨 AI生成結果")

        st.caption(f"使用API：{api_provider}")

        if st.session_state.generated_image_bytes is None:
            with st.spinner("AIがレンダリングしています..."):
                st.session_state.generated_image_bytes = generate_castle_image(
                    st.session_state.ai_input_image_bytes,
                    user_prompt,
                    api_provider,
                    selected_api_key
                )

        if st.session_state.generated_image_bytes:
            st.image(st.session_state.generated_image_bytes, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("別プロンプト・別APIで再生成", use_container_width=True):
            st.session_state.generated_image_bytes = None
            st.rerun()

    with col2:
        if st.button("最初からやり直す", use_container_width=True):
            reset_app()
