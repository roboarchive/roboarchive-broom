gstrain = gs://robo-broom/data/train

venv/bin/python:
	virtualenv --system-site-packages venv
	./venv/bin/pip install -r requirements.txt

sync_local:
	mkdir -p train
	gsutil -m rsync -r $(gstrain) train

sync_remote:
	gsutil -m rsync -r train $(gstrain)

force_sync_local:
	mkdir -p train
	gsutil -m rsync -d -r $(gstrain) train

force_sync_remote:
	gsutil -m rsync -d -r train $(gstrain)
