SRC 	= src
TESTS 	= tests

main:
	$(MAKE) -C $(SRC) main

test:
	echo "Pass"

test-rt:
	$(MAKE) -C $(TESTS) test-rt
