# Serving manifests

`build_manifest.py` turns an edition spec plus the upload-time weights manifest into
the manifest base the runtime signs. `editions/` holds one spec per Lebrel edition;
`out/` holds generated manifests (committed so reviewers can diff them).

Production edition (DeepSeek V4 Flash Uncensored):

    python3 -m manifest.build_manifest \
        --edition manifest/editions/lebrel-deepseek-v4-flash-uncensored.json \
        --weights-manifest ../modal-deepseek-encrypted/model_manifest.json \
        --out manifest/out/lebrel-deepseek-v4-flash-uncensored.manifest.json

The weights manifest is the file whose SHA-256 the Modal service pins
(`MODEL_MANIFEST_SHA256`), so the signed manifest describes exactly the bytes the
runtime loads. The generated file is what goes into `LEBREL_SERVING_MANIFEST_JSON`.
