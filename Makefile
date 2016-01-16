.PHONY: clean

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.pyx.md5" | xargs rm -f

ensemble_model:
	cd utils/scripts && python ensemble_classifier_script.py

random_forest:
	cd utils/scripts && python random_forest_script.py