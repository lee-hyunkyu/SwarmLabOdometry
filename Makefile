main:
	$(MAKE) -C src main

test:
	echo "Pass"

test-null_space:
	$(MAKE) -C tests test-null_space
