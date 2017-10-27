# simulator

## usage
** port_setting **

	set agent and user communicate port, default is 8080 and 8081
	set slots, default is 5 slots, delete some when needed

** run.sh **

	make sure you have python 3
	default command is "python3 xxx.py" or you need to change "python3" to "your py3 cmd" in run.sh
	
** test_client.py **

	for test simulator
	test should after let run.sh running
	2 mode: auto or human input
	in human input: 
		action command: new_dialogue is used for start a new dialogue; action is used for input actions based on now dialogue
		ask or confirm command: you have six choices: time/duration/money/location/number/nothing