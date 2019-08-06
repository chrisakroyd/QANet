from src import config, constants, util
from train import train
from preprocess import preprocess
from test import test
from demo import demo
from download import download
from ensemble import ensemble


def main(sess_config, orig_params):
    if orig_params.help:
        print(orig_params)
        exit(0)

    mode = orig_params.mode.lower().strip()

    # If we are in modes other than download / pre-process load pre-existing configs.
    if mode in {constants.Modes.TRAIN, constants.Modes.TEST, constants.Modes.DEMO, constants.Modes.DEBUG,
                constants.Modes.CHECKPOINT_ENSEMBLE}:
        params = util.load_config(orig_params, util.config_path(orig_params))  # Loads a pre-existing config otherwise == params
    else:
        params = orig_params

    # Some parameters in ensemble mode always need to be taken from the command line.
    if mode == constants.Modes.ENSEMBLE or mode == constants.Modes.CHECKPOINT_ENSEMBLE:
        params.gradual = orig_params.gradual
        params.max_models = orig_params.max_models

    if mode == constants.Modes.TRAIN:
        train(sess_config, params)
    elif mode == constants.Modes.DEBUG:
        train(sess_config, params, debug=True)
    elif mode == constants.Modes.PREPROCESS:
        preprocess(params)
    elif mode == constants.Modes.TEST:
        test(sess_config, params)
    elif mode == constants.Modes.CHECKPOINT_SELECTION:
        test(sess_config, params, checkpoint_selection=True)
    elif mode == constants.Modes.CHECKPOINT_ENSEMBLE:
        ensemble(sess_config, params, checkpoint_ensemble=True)
    elif mode == constants.Modes.ENSEMBLE:
        ensemble(sess_config, params)
    elif mode == constants.Modes.DEMO:
        app = demo(sess_config, params)
        app.run(port=5000)
    elif mode == constants.Modes.DOWNLOAD:
        download(params)
    else:
        print('Unknown Mode: {}'.format(mode))
        exit(0)


if __name__ == '__main__':
    defaults = util.namespace_json(path=constants.FilePaths.DEFAULTS)
    main(config.gpu_config(), config.model_config(defaults).FLAGS)
