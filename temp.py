from transformers import AutoTokenizer, XCLIPTextModel

model = XCLIPTextModel.from_pretrained("microsoft/xclip-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")