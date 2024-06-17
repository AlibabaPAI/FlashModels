.PHONY: format

format:
	isort flashmodels/ apps/ tools/ examples/
	yapf -i -r *.py flashmodels/ apps/ tools/ examples/
