"""Export fixed-seed, untrained encoders without presenting them as final pretrained artifacts."""
import json
from common.identity import semantic_identity, tensor_state_hash
from common.io import atomic_torch_save
from common.outputs import open_run_directory
from common.training import seed_everything
from stage1.data import PreparedCorpusDataset
from stage1.identity import metadata_identity, validate_feature_generation_runtime
from stage1.model import build_stage1_model
from stage1.tokenizer import SmilesTokenizer


def build_random_encoder_payload(config):
    if not config.is_dual_view or config.model_factory is not None:
        raise ValueError('Random encoder export requires the v4 dual-view architecture')
    root = config.data.artifacts_dir
    metadata = json.loads((root / 'metadata.json').read_text())
    validate_feature_generation_runtime(metadata)
    corpus = metadata_identity(metadata, 'corpus', context='Random Stage1 corpus')
    feature = metadata_identity(metadata, 'feature', context='Random Stage1 features')
    dataset = PreparedCorpusDataset(root, 'train', config.data.shard_cache_size)
    vocabulary = SmilesTokenizer.load(root / 'tokenizer.json')
    seed_everything(config.data.seed)
    model = build_stage1_model(config, vocabulary, dataset.descriptor_schema, encoder_only=True).requires_grad_(False).eval()
    state = model.state_dict()
    state_hash = tensor_state_hash('stage1.dual-view.encoder.v4', state)
    identity = semantic_identity('ablation.stage1.random-encoder.v1', {
        'config': config.to_dict(), 'random_seed': config.data.seed,
        'corpus_identity': corpus['hash'], 'feature_identity': feature['hash'],
        'state_hash': state_hash, 'optimizer_updates': 0,
    })
    return {'kind': 'ilume_stage1_random_encoder_v4', 'format_version': 4,
            'config': config.to_dict(), 'model': state, 'state_hash': state_hash,
            'training_identity': identity, 'corpus_identity': corpus, 'fixed_final_epoch': 0}


def export_random_encoder(config, config_path, output):
    payload = build_random_encoder_payload(config)
    identity, state_hash = payload['training_identity'], payload['state_hash']
    run = open_run_directory(stage='stage1', operation='train', config_path=config_path,
        config_payload=config.to_dict(), output=output, seed=config.data.seed,
        semantic_identity=identity, details={'initialization': 'random_frozen', 'optimizer_updates': 0})
    try:
        artifact = run.root / 'stage1_encoder.pt'
        atomic_torch_save(artifact, payload)
        from common.io import sha256_file
        run.complete({'optimizer_updates': 0, 'state_hash': state_hash,
                      'artifact': artifact.name, 'artifact_sha256': sha256_file(artifact)})
    except BaseException:
        run.fail()
        raise
