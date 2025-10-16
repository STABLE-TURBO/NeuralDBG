import importlib, sys
mods=['click','lark','numpy','plotly','torch','tensorflow','onnx']
results=[]
for m in mods:
    try:
        importlib.import_module(m)
        results.append(f"{m}:OK")
    except Exception as e:
        results.append(f"{m}:ERR:{e.__class__.__name__}:{e}")
open('import_check.txt','w',encoding='utf-8').write('\n'.join(results))

