# FlowMLLab blind-case explorer

This small Streamlit application makes the retained POD--DeepONet evidence
inspectable without retraining a model. The selector contains only the three
retained test Reynolds-number cases in the v1.1.0 archive. The updated
multi-output POD--DeepONet predicts both velocity and zero-mean pressure. A
separate pressure trunk and branch prevent pressure scaling from changing the
divergence-free velocity basis. The pressure panel therefore shows a direct
learned output, while still making clear that its CFD labels were originally
obtained with the declared momentum-gradient least-squares procedure.

## Run locally

```bash
python -m pip install -e ".[demo]"
streamlit run demo/streamlit_app.py
```

The application reads the versioned files under `data/` and
`results/pod_deeponet/`. It does not write results or change the scientific
protocol.

## Deploy for public use

On Streamlit Community Cloud, select this public repository and use
`demo/streamlit_app.py` as the entry point. The lightweight
`demo/requirements.txt` is placed beside the application so the deployment does
not install the optional TensorFlow teaching environment.

After deployment, replace the README's repository-local **Explore the blind
demo** link with the public application URL.
