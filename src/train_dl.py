def save_dl_model(model, path="models/model_mlp.h5"):
    model.save(path)
    print(f"Deep Learning model saved to: {path}")
