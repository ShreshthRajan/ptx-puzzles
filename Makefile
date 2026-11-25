P ?=

.PHONY: grade solutions validate
grade:
	python grade.py $(P)
solutions:
	python grade.py --dir solutions $(P)
validate:
	python grade.py --validate $(P)
