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

build_docker:
	docker build -t 'roboarchive/runner:2' -f infra/Dockerfile .

push_docker:
	docker push 'roboarchive/runner:2'

label_clean_train:
	PYTHONPATH=. python3 ./cleaning_tool/train_cnn.py -b 1 --epoch-steps 5 --cpu -e 10 --period=30 -d --tile-size=256
	echo 'ok'
