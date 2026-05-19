cleanall:
	rm -rf runs/exp*/output/* runs/exp*/batch_*

# generate data for oscNet training
oscNetData:
	rm -rf data/*
	python src/oscillator_data_generation/generate_data.py

inspdataloader:
	python src/oscillatorNet/data_loader.py

trainOscNet:
	python src/oscillatorNet/train.py