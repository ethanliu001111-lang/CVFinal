#!/usr/bin/env python3
"""One-shot: extract a clean 5-slide template from any prior build of Intro to CV.pptx.

Reads the current pptx, identifies the 5 template slides (title, 3
TITLE_AND_BODY section dividers, Thanks), drops everything else from the
zip (orphan slides + their rels), rewrites presentation.xml + its rels +
[Content_Types].xml to enumerate only those 5 slides, saves the result to
`presentation/template.pptx`. The build script then uses that as a stable
input.

Run once after the deck is in a known-good state. Re-run only if the
template structure changes.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "presentation" / "Intro to CV.pptx"
DST = ROOT / "presentation" / "template.pptx"

P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"

ET.register_namespace("", P_NS)
ET.register_namespace("r", R_NS)


def main():
    # 1. Open and read.
    with zipfile.ZipFile(SRC, "r") as zin:
        all_files = {info.filename: zin.read(info.filename) for info in zin.infolist()}

    # 2. Inspect each slide to figure out which to keep (template-typed).
    pres_xml = ET.fromstring(all_files["ppt/presentation.xml"])
    sldIdLst = pres_xml.find(f"{{{P_NS}}}sldIdLst")
    sld_ids = sldIdLst.findall(f"{{{P_NS}}}sldId")
    rId_attr = f"{{{R_NS}}}id"

    rels_xml = ET.fromstring(all_files["ppt/_rels/presentation.xml.rels"])
    rid_to_target = {}
    for rel in rels_xml.findall(f"{{{REL_NS}}}Relationship"):
        rid_to_target[rel.get("Id")] = (rel.get("Target"), rel.get("Type"))

    # Decide which slides are "template": we know the original template was
    # title (idx 0) → RW divider → PR divider → AN divider → Thanks (idx 4).
    # In the current file these are present in order at known partnames.
    # The simplest robust rule: pick the first sldId for each layout we want.
    # But we'd need to inspect each slide's layout. Simpler still: keep the
    # first 5 sldIds whose layouts match TITLE / TITLE_AND_BODY (×3) / TITLE_1_1_1_1.

    keep_sld_ids: list = []
    layouts_seen = {"title": 0, "body": 0, "thanks": 0}
    template_layout_targets = {
        "TITLE": "title", "TITLE_AND_BODY": "body", "TITLE_1_1_1_1": "thanks",
    }

    def slide_layout_name(slide_xml_path: str) -> str:
        # slide rels reference its slideLayout
        rels_path = slide_xml_path.replace("slides/", "slides/_rels/") + ".rels"
        if rels_path not in all_files:
            return ""
        rels = ET.fromstring(all_files[rels_path])
        for rel in rels.findall(f"{{{REL_NS}}}Relationship"):
            if "slideLayout" in rel.get("Type", ""):
                layout_target = rel.get("Target")  # e.g. "../slideLayouts/slideLayout1.xml"
                # Find the layout file content to read its name attr
                layout_name_path = "ppt/" + layout_target.lstrip("./").replace("../", "")
                if layout_name_path in all_files:
                    layout_xml = ET.fromstring(all_files[layout_name_path])
                    cSld = layout_xml.find(f"{{{P_NS}}}cSld")
                    if cSld is not None:
                        return cSld.get("name", "")
        return ""

    for sld_id_el in sld_ids:
        rid = sld_id_el.get(rId_attr)
        target, _ = rid_to_target[rid]
        slide_path = "ppt/" + target
        layout = slide_layout_name(slide_path)
        target_kind = template_layout_targets.get(layout)
        if target_kind == "title" and layouts_seen["title"] == 0:
            keep_sld_ids.append((sld_id_el, rid, slide_path)); layouts_seen["title"] += 1
        elif target_kind == "body" and layouts_seen["body"] < 3:
            keep_sld_ids.append((sld_id_el, rid, slide_path)); layouts_seen["body"] += 1
        elif target_kind == "thanks" and layouts_seen["thanks"] == 0:
            keep_sld_ids.append((sld_id_el, rid, slide_path)); layouts_seen["thanks"] += 1
        if len(keep_sld_ids) == 5:
            break

    if len(keep_sld_ids) != 5:
        raise RuntimeError(f"could not find 5 template slides; only {len(keep_sld_ids)} matched")

    print(f"Keeping {len(keep_sld_ids)} slides:")
    for sld_id_el, rid, path in keep_sld_ids:
        print(f"   {rid}  ·  {path}")

    keep_paths = {p for _, _, p in keep_sld_ids}
    keep_rids = {rid for _, rid, _ in keep_sld_ids}

    # 3. Rewrite sldIdLst — keep only the 5 we picked, in order.
    for child in list(sldIdLst):
        sldIdLst.remove(child)
    for sld_id_el, _, _ in keep_sld_ids:
        sldIdLst.append(sld_id_el)
    new_pres_xml = ET.tostring(pres_xml, xml_declaration=True, encoding="UTF-8")

    # 4. Rewrite presentation.xml.rels — drop slide rels not in keep_rids.
    for rel in list(rels_xml):
        rid = rel.get("Id")
        rel_type = rel.get("Type", "")
        if "/slide" in rel_type and rel_type.endswith("/slide"):
            if rid not in keep_rids:
                rels_xml.remove(rel)
    new_rels_xml = ET.tostring(rels_xml, xml_declaration=True, encoding="UTF-8")

    # 5. Rewrite [Content_Types].xml — drop slide overrides for orphan slides.
    ct_xml = ET.fromstring(all_files["[Content_Types].xml"])
    for ov in list(ct_xml):
        if ov.tag.endswith("Override"):
            part = ov.get("PartName", "")
            if part.startswith("/ppt/slides/slide") and part.endswith(".xml"):
                if part.lstrip("/") not in keep_paths:
                    ct_xml.remove(ov)
    new_ct_xml = ET.tostring(ct_xml, xml_declaration=True, encoding="UTF-8")

    # 6. Build the output zip — drop orphan slide xmls and their rels files.
    drop_prefixes = []
    keep_slide_basenames = {p.split("/")[-1] for p in keep_paths}
    drop_files = set()
    for name in all_files:
        if name.startswith("ppt/slides/slide") and name.endswith(".xml"):
            base = name.split("/")[-1]
            if base not in keep_slide_basenames:
                drop_files.add(name)
        if name.startswith("ppt/slides/_rels/slide") and name.endswith(".xml.rels"):
            base = name.split("/")[-1].replace(".rels", "")
            if base not in keep_slide_basenames:
                drop_files.add(name)

    print(f"Dropping {len(drop_files)} orphan files from the package")

    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as zout:
        written = set()
        for name, content in all_files.items():
            if name in drop_files or name in written:
                continue
            if name == "ppt/presentation.xml":
                content = new_pres_xml
            elif name == "ppt/_rels/presentation.xml.rels":
                content = new_rels_xml
            elif name == "[Content_Types].xml":
                content = new_ct_xml
            zout.writestr(name, content)
            written.add(name)

    print(f"✓ wrote {DST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
