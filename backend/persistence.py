"""Explicit local file association and atomic model saves."""
import hashlib
import io
import os
import re
from pathlib import Path
import tempfile
import zipfile

import mujoco as mj
from lxml import etree


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def bind_source(editor, path):
    editor.source_path = str(path)
    editor.source_digest = digest(path)


def destination(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("请填写本机绝对路径")
    path = path.resolve()
    if path.suffix.lower() not in (".xml", ".mjcfproj"):
        raise ValueError("保存位置必须是 .xml 或 .mjcfproj 文件")
    if not path.parent.is_dir():
        raise ValueError("保存目录不存在")
    return path


def atomic_write(path, data):
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            os.chmod(temporary, path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def readable_assets(xml, assets, directory=None):
    """Name resources by their first owning joint/body; never overwrite other bytes."""
    root = etree.fromstring(xml.encode() if isinstance(xml, str) else xml)
    owners = {}
    for body in root.iter('body'):
        joints = body.findall('joint') + body.findall('freejoint')
        joint = next((j.get('name') for j in joints if j.get('name')), 'fixed')
        for geom in body.findall('geom'):
            mesh = geom.get('mesh')
            if mesh and mesh not in owners:
                owners[mesh] = joint + '_' + body.get('name', 'link')
    bases = {}
    for node in root.findall('./asset/*'):
        label = owners.get(node.get('name')) if node.tag == 'mesh' else None
        for attr, value in node.attrib.items():
            if (attr == 'file' or attr.startswith('file')) and value in assets:
                bases.setdefault(value, label or node.get('name') or node.tag)
    mapping, renamed = {}, {}
    for original, data in assets.items():
        stem = re.sub(r'[^\w.-]+', '_', bases.get(original, Path(original).stem)).strip('._')[:120] or 'asset'
        suffix = Path(original).suffix
        number = 1
        while True:
            name = 'assets/' + stem + ('' if number == 1 else '_' + str(number)) + suffix
            existing = renamed.get(name)
            path = directory / name if directory is not None else None
            conflict = (existing is not None and existing != data) or (path is not None and path.exists() and path.read_bytes() != data)
            if not conflict:
                break
            number += 1
        mapping[original] = name
        renamed[name] = data
    for node in root.iter():
        for attr, value in list(node.attrib.items()):
            if attr.startswith('file') and value in mapping:
                node.set(attr, mapping[value])
    return etree.tostring(root, pretty_print=True), renamed


def save_local(editor, value=None):
    path = destination(value or editor.source_path or '')
    expected = editor.source_digest if str(path) == editor.source_path else digest(path)
    if digest(path) != expected:
        raise ValueError("原文件已被外部修改，未覆盖。请重新打开或另存到其他路径。")
    assets = {}
    if path.suffix.lower() == '.mjcfproj':
        content = editor.archive(True)
    else:
        archive = zipfile.ZipFile(io.BytesIO(editor.archive(False)))
        raw_assets = {item.filename: archive.read(item.filename)
                      for item in archive.infolist() if item.filename != 'model.xml'}
        content, assets = readable_assets(archive.read('model.xml'), raw_assets, path.parent)
        mj.MjSpec.from_string(content.decode(), assets=assets).compile()
    # Reuse identical resources; distinct existing contents get a numbered filename.
    for name, data in assets.items():
        output = path.parent / name
        if output.exists() and output.read_bytes() != data:
            raise ValueError("资源文件冲突，未覆盖: " + str(output))
    created = []
    try:
        for name, data in assets.items():
            output = path.parent / name
            if not output.exists():
                output.parent.mkdir(exist_ok=True)
                atomic_write(output, data)
                created.append(output)
        if digest(path) != expected:
            raise ValueError("保存期间原文件发生外部修改，未覆盖")
        atomic_write(path, content)
    except Exception:
        for output in created:
            output.unlink(missing_ok=True)
        raise
    bind_source(editor, path)
    return {'path': str(path), 'revision': editor.revision}
