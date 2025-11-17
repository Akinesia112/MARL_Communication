# assets/make_four_ants_xml.py
import os, re
import xml.etree.ElementTree as ET
from gymnasium.envs.mujoco.ant_v5 import ASSET_PATH

ANT_XML = os.path.join(ASSET_PATH, "ant.xml")

def _prefix_names(elem, prefix):
    """遞迴把 body/geom/joint/site 等常見 name 屬性加前綴，並修正 actuator/joint 引用。"""
    name_attrs = {"body","geom","joint","site","camera","light","tendon","equality","sensor"}
    # 先處理自己
    if "name" in elem.attrib and elem.tag in name_attrs:
        elem.attrib["name"] = f"{prefix}{elem.attrib['name']}"
    # 修正會引用 'joint' 的屬性
    if "joint" in elem.attrib:
        elem.attrib["joint"] = f"{prefix}{elem.attrib['joint']}"
    if "body" in elem.attrib:
        elem.attrib["body"] = f"{prefix}{elem.attrib['body']}"
    if "site" in elem.attrib:
        elem.attrib["site"] = f"{prefix}{elem.attrib['site']}"
    if "tendon" in elem.attrib:
        elem.attrib["tendon"] = f"{prefix}{elem.attrib['tendon']}"
    # 遞迴
    for c in list(elem):
        _prefix_names(c, prefix)

def make_four_ants_xml(out_path="assets/four_ants.xml", spacing=5.0):
    """
    讀 ant.xml，抽出 worldbody 下的唯一一個主體 body（Ant torso），複製四份。
    也把 <actuator> 中所有 motor 複製 4 份並改 joint 引用。
    """
    tree = ET.parse(ANT_XML); root = tree.getroot()
    worldbody = root.find("worldbody")
    actuators = root.find("actuator")
    if worldbody is None or actuators is None:
        raise RuntimeError("unexpected ant.xml structure")

    # 找出 ant 的主 body（torso）
    ant_body = None
    for child in worldbody:
        if child.tag == "body":
            ant_body = child
            break
    if ant_body is None:
        raise RuntimeError("torso body not found in ant.xml")

    # 預先把原始 actuator 的 motor 節點抓出來
    base_motors = [m for m in list(actuators) if m.tag == "motor"]

    # 移除原本單隻 ant 的 body 與 actuators，等會兒用複製的取代
    worldbody.remove(ant_body)
    for m in base_motors:
        actuators.remove(m)

    # 四個起始位置（你可照需求調整）
    positions = [(0,0), (spacing,0), (0,spacing), (spacing,spacing)]
    prefixes  = ["a1/", "a2/", "a3/", "a4/"]

    for (px, py), pref in zip(positions, prefixes):
        # 複製身體 subtree
        b = ET.fromstring(ET.tostring(ant_body))
        _prefix_names(b, pref)
        # 在根 body 平移放置
        b.attrib["pos"] = f"{px} {py} 0"
        worldbody.append(b)

        # 複製所有 motor，修 joint 指向前綴後的名稱，motor name 也加前綴
        for m in base_motors:
            mm = ET.fromstring(ET.tostring(m))
            if "name" in mm.attrib:
                mm.attrib["name"] = pref + mm.attrib["name"]
            if "joint" in mm.attrib:
                mm.attrib["joint"] = pref + mm.attrib["joint"]
            actuators.append(mm)

    # 可選：把地面做大、加點光源等（非必要）
    ET.indent(root)  # Python 3.9+
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"[four-ants] wrote {out_path}")

if __name__ == "__main__":
    make_four_ants_xml()
