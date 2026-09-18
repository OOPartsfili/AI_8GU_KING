"""Generate a dependency-free Xcode project, shared app scheme and UI tests."""
from pathlib import Path
import hashlib
import json

IOS=Path(__file__).resolve().parents[1]


def oid(name):
    return hashlib.sha1(name.encode()).hexdigest()[:24].upper()


def encode(value, level=0):
    indent='\t'*level
    if isinstance(value,dict):
        return '{\n'+''.join(indent+'\t'+json.dumps(str(key))+' = '+encode(item,level+1)+';\n' for key,item in value.items())+indent+'}'
    if isinstance(value,list):
        return '(\n'+''.join(indent+'\t'+encode(item,level+1)+',\n' for item in value)+indent+')'
    return str(value) if isinstance(value,int) else json.dumps(value,ensure_ascii=False)


def generate():
    objects={}
    def add(object_key,isa,**fields):
        key=oid(object_key);objects[key]=dict(isa=isa,**fields);return key
    app_sources=[];app_files=[]
    for file in sorted((IOS/'AI8GUKing').glob('*.swift')):
        reference=add('file:'+file.name,'PBXFileReference',lastKnownFileType='sourcecode.swift',path=file.name,sourceTree='<group>')
        app_files.append(reference);app_sources.append(add('build:'+file.name,'PBXBuildFile',fileRef=reference))
    info=add('file:Info','PBXFileReference',lastKnownFileType='text.plist.xml',path='Info.plist',sourceTree='<group>')
    # "Resources" is reserved by CFBundle layout detection; use an app-specific folder.
    resources=add('file:Resources','PBXFileReference',lastKnownFileType='folder',path='OfflineContent',sourceTree='<group>')
    assets=add('file:Assets','PBXFileReference',lastKnownFileType='folder.assetcatalog',path='Assets.xcassets',sourceTree='<group>')
    resource_builds=[add('build:Resources','PBXBuildFile',fileRef=resources),add('build:Assets','PBXBuildFile',fileRef=assets)]
    app_group=add('group:app','PBXGroup',children=app_files+[info,resources,assets],path='AI8GUKing',sourceTree='<group>')
    ui_file=add('file:UITests','PBXFileReference',lastKnownFileType='sourcecode.swift',path='AI8GUKingUITests.swift',sourceTree='<group>')
    ui_group=add('group:uitests','PBXGroup',children=[ui_file],path='AI8GUKingUITests',sourceTree='<group>')
    ui_source=add('build:UITests','PBXBuildFile',fileRef=ui_file)
    unit_file=add('file:UnitTests','PBXFileReference',lastKnownFileType='sourcecode.swift',path='OfflineRuntimeTests.swift',sourceTree='<group>')
    unit_group=add('group:unittests','PBXGroup',children=[unit_file],path='AI8GUKingTests',sourceTree='<group>')
    unit_source=add('build:UnitTests','PBXBuildFile',fileRef=unit_file)
    app_product=add('product:app','PBXFileReference',explicitFileType='wrapper.application',includeInIndex=0,path='AI8GUKing.app',sourceTree='BUILT_PRODUCTS_DIR')
    ui_product=add('product:uitests','PBXFileReference',explicitFileType='wrapper.cfbundle',includeInIndex=0,path='AI8GUKingUITests.xctest',sourceTree='BUILT_PRODUCTS_DIR')
    unit_product=add('product:unittests','PBXFileReference',explicitFileType='wrapper.cfbundle',includeInIndex=0,path='AI8GUKingTests.xctest',sourceTree='BUILT_PRODUCTS_DIR')
    products=add('group:products','PBXGroup',children=[app_product,ui_product,unit_product],name='Products',sourceTree='<group>')
    main_group=add('group:main','PBXGroup',children=[app_group,ui_group,unit_group,products],sourceTree='<group>')
    def phase(name,isa,files):
        return add(name,isa,buildActionMask=2147483647,files=files,runOnlyForDeploymentPostprocessing=0)
    sources=phase('phase:sources','PBXSourcesBuildPhase',app_sources)
    frameworks=phase('phase:frameworks','PBXFrameworksBuildPhase',[])
    resource_phase=phase('phase:resources','PBXResourcesBuildPhase',resource_builds)
    ui_sources=phase('phase:ui_sources','PBXSourcesBuildPhase',[ui_source])
    ui_frameworks=phase('phase:ui_frameworks','PBXFrameworksBuildPhase',[])
    ui_resources=phase('phase:ui_resources','PBXResourcesBuildPhase',[])
    unit_sources=phase('phase:unit_sources','PBXSourcesBuildPhase',[unit_source])
    unit_frameworks=phase('phase:unit_frameworks','PBXFrameworksBuildPhase',[])
    unit_resources=phase('phase:unit_resources','PBXResourcesBuildPhase',[])
    project_settings=dict(ALWAYS_SEARCH_USER_PATHS='NO',CLANG_ENABLE_MODULES='YES',CLANG_ENABLE_OBJC_ARC='YES',
                          IPHONEOS_DEPLOYMENT_TARGET='17.0',SDKROOT='iphoneos',SWIFT_VERSION='5.0',
                          ENABLE_USER_SCRIPT_SANDBOXING='YES',GCC_C_LANGUAGE_STANDARD='gnu17')
    app_settings=dict(PRODUCT_BUNDLE_IDENTIFIER='com.oopartsfili.ai8guking',PRODUCT_NAME='$(TARGET_NAME)',
                      INFOPLIST_FILE='AI8GUKing/Info.plist',GENERATE_INFOPLIST_FILE='NO',
                      TARGETED_DEVICE_FAMILY='1',SUPPORTED_PLATFORMS='iphoneos iphonesimulator',
                      SUPPORTS_MACCATALYST='NO',CODE_SIGN_STYLE='Automatic',
                      ASSETCATALOG_COMPILER_APPICON_NAME='AppIcon',MARKETING_VERSION='1.5.0',CURRENT_PROJECT_VERSION='1',
                      LD_RUNPATH_SEARCH_PATHS=['$(inherited)','@executable_path/Frameworks'])
    ui_settings=dict(PRODUCT_BUNDLE_IDENTIFIER='com.oopartsfili.ai8guking.uitests',PRODUCT_NAME='$(TARGET_NAME)',
                     GENERATE_INFOPLIST_FILE='YES',TARGETED_DEVICE_FAMILY='1',TEST_TARGET_NAME='AI8GUKing',
                     CODE_SIGN_STYLE='Automatic',LD_RUNPATH_SEARCH_PATHS=['$(inherited)','@executable_path/Frameworks','@loader_path/Frameworks'])
    unit_settings=dict(ui_settings,PRODUCT_BUNDLE_IDENTIFIER='com.oopartsfili.ai8guking.tests',
                       TEST_HOST='$(BUILT_PRODUCTS_DIR)/AI8GUKing.app/$(BUNDLE_EXECUTABLE_FOLDER_PATH)/AI8GUKing',BUNDLE_LOADER='$(TEST_HOST)')
    def configs(name,settings):
        ids=[]
        for config in ('Debug','Release'):
            values=dict(settings)
            if config=='Debug':
                values.update(SWIFT_OPTIMIZATION_LEVEL='-Onone',SWIFT_ACTIVE_COMPILATION_CONDITIONS='DEBUG',ENABLE_TESTABILITY='YES')
            else:
                values.update(SWIFT_COMPILATION_MODE='wholemodule',SWIFT_OPTIMIZATION_LEVEL='-O',DEBUG_INFORMATION_FORMAT='dwarf-with-dsym')
            ids.append(add('config:'+name+config,'XCBuildConfiguration',buildSettings=values,name=config))
        return add('configs:'+name,'XCConfigurationList',buildConfigurations=ids,defaultConfigurationIsVisible=0,defaultConfigurationName='Release')
    project_configs=configs('project',project_settings)
    app_configs=configs('app',app_settings)
    ui_configs=configs('uitests',ui_settings)
    unit_configs=configs('unittests',unit_settings)
    app=add('target:app','PBXNativeTarget',buildConfigurationList=app_configs,buildPhases=[sources,frameworks,resource_phase],
            buildRules=[],dependencies=[],name='AI8GUKing',productName='AI8GUKing',productReference=app_product,productType='com.apple.product-type.application')
    proxy=add('proxy:app','PBXContainerItemProxy',containerPortal=oid('project'),proxyType=1,remoteGlobalIDString=app,remoteInfo='AI8GUKing')
    dependency=add('dependency:app','PBXTargetDependency',target=app,targetProxy=proxy)
    ui=add('target:uitests','PBXNativeTarget',buildConfigurationList=ui_configs,buildPhases=[ui_sources,ui_frameworks,ui_resources],
           buildRules=[],dependencies=[dependency],name='AI8GUKingUITests',productName='AI8GUKingUITests',productReference=ui_product,productType='com.apple.product-type.bundle.ui-testing')
    unit=add('target:unittests','PBXNativeTarget',buildConfigurationList=unit_configs,buildPhases=[unit_sources,unit_frameworks,unit_resources],
             buildRules=[],dependencies=[dependency],name='AI8GUKingTests',productName='AI8GUKingTests',productReference=unit_product,productType='com.apple.product-type.bundle.unit-test')
    project=add('project','PBXProject',attributes={'BuildIndependentTargetsInParallel':'YES','LastUpgradeCheck':'1640',
                'TargetAttributes':{app:{'CreatedOnToolsVersion':'16.4'},ui:{'CreatedOnToolsVersion':'16.4','TestTargetID':app},unit:{'CreatedOnToolsVersion':'16.4','TestTargetID':app}}},
                buildConfigurationList=project_configs,compatibilityVersion='Xcode 14.0',developmentRegion='zh_CN',
                hasScannedForEncodings=0,knownRegions=['en','zh-Hans','Base'],mainGroup=main_group,productRefGroup=products,
                projectDirPath='',projectRoot='',targets=[app,ui,unit])
    directory=IOS/'AI8GUKing.xcodeproj';directory.mkdir(exist_ok=True)
    payload=dict(archiveVersion=1,classes={},objectVersion=56,objects=objects,rootObject=project)
    (directory/'project.pbxproj').write_bytes(('// !$*UTF8*$!\n'+encode(payload)+'\n').encode('utf-8'))
    schemes=directory/'xcshareddata/xcschemes';schemes.mkdir(parents=True,exist_ok=True)
    def reference(identifier,name):
        return f'<BuildableReference BuildableIdentifier="primary" BlueprintIdentifier="{identifier}" BuildableName="{name}" BlueprintName="{name.split(".")[0]}" ReferencedContainer="container:AI8GUKing.xcodeproj"/>'
    app_ref=reference(app,'AI8GUKing.app');ui_ref=reference(ui,'AI8GUKingUITests.xctest');unit_ref=reference(unit,'AI8GUKingTests.xctest')
    scheme=f'''<?xml version="1.0" encoding="UTF-8"?>
<Scheme LastUpgradeVersion="1640" version="1.3">
 <BuildAction parallelizeBuildables="YES" buildImplicitDependencies="YES"><BuildActionEntries>
  <BuildActionEntry buildForTesting="YES" buildForRunning="YES" buildForProfiling="YES" buildForArchiving="YES" buildForAnalyzing="YES">{app_ref}</BuildActionEntry>
  <BuildActionEntry buildForTesting="YES" buildForRunning="NO" buildForProfiling="NO" buildForArchiving="NO" buildForAnalyzing="YES">{ui_ref}</BuildActionEntry>
  <BuildActionEntry buildForTesting="YES" buildForRunning="NO" buildForProfiling="NO" buildForArchiving="NO" buildForAnalyzing="YES">{unit_ref}</BuildActionEntry>
 </BuildActionEntries></BuildAction>
 <TestAction buildConfiguration="Debug" selectedDebuggerIdentifier="Xcode.DebuggerFoundation.Debugger.LLDB" selectedLauncherIdentifier="Xcode.IDEFoundation.Launcher.LLDB" shouldUseLaunchSchemeArgsEnv="YES"><Testables><TestableReference skipped="NO">{unit_ref}</TestableReference><TestableReference skipped="NO">{ui_ref}</TestableReference></Testables></TestAction>
 <LaunchAction buildConfiguration="Debug" selectedDebuggerIdentifier="Xcode.DebuggerFoundation.Debugger.LLDB" selectedLauncherIdentifier="Xcode.IDEFoundation.Launcher.LLDB" launchStyle="0" useCustomWorkingDirectory="NO" ignoresPersistentStateOnLaunch="NO" debugDocumentVersioning="YES" debugServiceExtension="internal" allowLocationSimulation="YES"><BuildableProductRunnable runnableDebuggingMode="0">{app_ref}</BuildableProductRunnable></LaunchAction>
 <ProfileAction buildConfiguration="Release" shouldUseLaunchSchemeArgsEnv="YES" savedToolIdentifier="" useCustomWorkingDirectory="NO" debugDocumentVersioning="YES"><BuildableProductRunnable runnableDebuggingMode="0">{app_ref}</BuildableProductRunnable></ProfileAction>
 <AnalyzeAction buildConfiguration="Debug"/><ArchiveAction buildConfiguration="Release" revealArchiveInOrganizer="YES"/>
</Scheme>'''
    (schemes/'AI8GUKing.xcscheme').write_bytes(scheme.encode())
    print('Generated',directory)


if __name__=='__main__':generate()
