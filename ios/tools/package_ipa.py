"""Package a real iphoneos arm64 app for Windows re-signing, never a simulator app."""
from pathlib import Path
import argparse
import hashlib
import json
import plistlib
import subprocess
import zipfile


def package(app, output):
    info=plistlib.loads((app/'Info.plist').read_bytes())
    assert info['CFBundleSupportedPlatforms']==['iPhoneOS'], 'A simulator build cannot be installed on iPhone'
    binary=app/info['CFBundleExecutable']
    architecture=subprocess.check_output(['lipo','-archs',str(binary)],text=True).strip()
    assert 'arm64' in architecture.split(), architecture
    assert not (app/'Resources').exists(), 'Reserved Resources directory can break iOS bundle detection'
    manifest=json.loads((app/'OfflineContent/content-manifest.json').read_text())
    reader=(app/'OfflineContent/reader.html').read_bytes()
    assert hashlib.sha256(reader).hexdigest()==manifest['reader_sha256']
    output.mkdir(parents=True,exist_ok=True)
    ipa=output/'AI8GUKing-1.5.0-unsigned.ipa'
    with zipfile.ZipFile(ipa,'w',zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(app.rglob('*')):
            if file.is_file():archive.write(file,Path('Payload')/app.name/file.relative_to(app))
    report=dict(file=ipa.name,sha256=hashlib.sha256(ipa.read_bytes()).hexdigest(),
                bytes=ipa.stat().st_size,bundle_id=info['CFBundleIdentifier'],architecture=architecture,
                supported_platforms=info['CFBundleSupportedPlatforms'],minimum_ios=info['MinimumOSVersion'],
                signing='unsigned — Windows re-signing required',content=manifest)
    (output/'package-manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--app',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();package(args.app,args.output)
