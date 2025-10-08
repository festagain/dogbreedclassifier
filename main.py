from telegram.ext import filters, CommandHandler, ContextTypes, Application, Updater, MessageHandler
from telegram import Update, error
from io import BytesIO
import PIL
import torch
from torchvision.transforms import transforms as T
from torch import device
import torchvision.models as models
import torch.nn as nn
import json
from tok import TOKEN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def initialize_model():

    base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for param in base.parameters():
        param.requires_grad = False

    in_features = base.fc.in_features
    base.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 120)

    )
    model = base.to(device)
    model.load_state_dict(torch.load("best_model.pth", weights_only=True, map_location=device))
    with open("ru_mapping.json", "r", encoding='utf-8') as f:
        mapping = json.load(f)
    return model, mapping


model, mapping = initialize_model()


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я бот, который умеет по картинке определить породу собаки!")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Отправь мне картинку собаки!")


# Responses

async def handle_message(update, context):
    await update.message.reply_text("Пожалуйста отправь картинку собаки!")



async def handle_photo(update, context):
    try:
        if update.message.photo:
            file = await context.bot.get_file(update.message.photo[-1].file_id)
            file_bytes = await file.download_as_bytearray()
        elif (update.message.document and update.message.document.mime_type and
              update.message.document.mime_type.startswith("image")):
            file = await context.bot.get_file(update.message.document.file_id)
            file_bytes = await file.download_as_bytearray()
        else:
            await update.message.reply_text("Я не нашёл изображение в сообщении"
                                            " — пришлите фото или файл-изображение.")
            return
    except error.TimedOut:
        await update.message.reply_text("Сервер Telegram слишком долго отвечает. "
                                        "Попробуйте отправить фото ещё раз.")
        return

    f = BytesIO(file_bytes)
    val_transforms  = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])
    sample = PIL.Image.open(f).convert("RGB")
    sample = val_transforms(sample).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(sample)
        probs = torch.softmax(output, dim=1)
        topk = torch.topk(probs, k = 3)
        top_vals = topk.values.squeeze(0).cpu().tolist()
        top_idx = topk.indices.squeeze(0).cpu().tolist()

    text = ["Топ 3 предположения: "]
    for idx, p in zip(top_idx, top_vals):
        label = mapping[str(idx)]
        text.append(f"{label} - {100.0 * p:.2f}%")
    await update.message.reply_text('\n'.join(text))

app = Application.builder().token(TOKEN).build()
app.add_handler(CommandHandler("start", start_command))
app.add_handler(CommandHandler("help", help_command))
app.add_handler(MessageHandler(filters.TEXT, handle_message))
app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
print("Polling")
app.run_polling()
