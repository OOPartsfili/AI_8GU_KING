"""Draw the app's original book/crown icon. Run locally with Pillow if changing it."""
from pathlib import Path
import json
from PIL import Image,ImageDraw

destination=Path(__file__).resolve().parents[1]/'AI8GUKing/Assets.xcassets/AppIcon.appiconset'
destination.mkdir(parents=True,exist_ok=True)
image=Image.new('RGB',(1024,1024),'#185e54');draw=ImageDraw.Draw(image)
# A compact open book and three-point crown; no external media or licensed fonts.
draw.rounded_rectangle((139,396,885,806),radius=60,fill='#f7f6ef')
draw.polygon([(176,427),(468,453),(510,504),(556,453),(848,427),(848,745),(552,774),(510,807),(468,774),(176,745)],fill='#e0ece1')
draw.polygon([(166,396),(453,423),(510,470),(568,423),(858,396),(858,704),(562,745),(510,790),(458,745),(166,704)],fill='#f9faf4')
draw.line([(510,481),(510,747)],fill='#185e54',width=15)
for left,right in [(225,433),(589,796)]:
    for y in (505,565,625):draw.rounded_rectangle((left,y,right,y+13),radius=6,fill='#aecabe')
draw.polygon([(340,342),(307,212),(425,259),(510,139),(597,259),(718,212),(683,342)],fill='#e9bd69')
draw.rounded_rectangle((340,354,683,379),radius=12,fill='#e9bd69')
image.save(destination/'AppIcon.png')
(destination/'Contents.json').write_text(json.dumps({'images':[{'filename':'AppIcon.png','idiom':'universal','platform':'ios','size':'1024x1024'}],
                                                   'info':{'author':'xcode','version':1}},indent=2)+'\n',encoding='utf-8')
print(destination/'AppIcon.png')
