from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO
from fastai.vision import *
import base64

# model_file_url = 'https://github.com/pankymathur/Fine-Grained-Clothing-Classification/blob/master/data/cloth_categories/models/stage-1_sz-150.pth?raw=true'
# model_file_name = 'stage-1-50-20.pth'
classes = ['plant', 'poison_oak']

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    # await download_file(model_file_url, path/'models'/f'{model_file_name}.pth')

    # path = '.'
    data = ImageDataBunch.from_folder("data", train=".", valid_pct=0.2, bs=8,
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)

    # data = ImageDataBunch.single_from_classes(path, classes, df_tfms=get_transforms(), size=672).normalize(imagenet_stats)
    # learn = create_cnn(data_bunch, models.resnet34, pretrained=False)
    learn = cnn_learner(data, models.resnet34, metrics=error_rate)
    # learn.load(model_file_name)
    learn.load('stage-1-34-5')
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

PREDICTION_FILE_SRC = path/'static'/'predictions.txt'

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    # breakpoint()
    img_bytes = data["img"]#await (data["img"])
    bytes = base64.b64decode(img_bytes)
    return predict_from_bytes(bytes)

def predict_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    _,_,losses = learn.predict(img)
    predictions = sorted(zip(classes, map(float, losses)), key=lambda p: p[1], reverse=True)
    result_html1 = path/'static'/'result1.html'
    result_html2 = path/'static'/'result2.html'
    
    result_html = str(result_html1.open().read() +str(predictions[0:3]) + result_html2.open().read())
    return HTMLResponse(result_html)

@app.route("/")
def form(request):
    index_html = path/'static'/'index.html'
    return HTMLResponse(index_html.open().read())

if __name__ == "__main__":
    if "serve" in sys.argv: uvicorn.run(app, host="0.0.0.0", port=8080)
