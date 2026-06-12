import sys


def load_conch_model(config):
    repo = config.model.conch_repo
    if repo and repo not in sys.path:
        sys.path.append(repo)
    from conch.open_clip_custom import create_model_from_pretrained

    model, preprocess = create_model_from_pretrained(config.model.encoder, config.model.conch_ckpt)
    return model, preprocess
