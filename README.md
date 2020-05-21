Edge Benchmarking Software
==========================

Prior to launching the benchmarking software, each model needs to be pre-partitioned to make benchmarking faster on the Edge. This can be done using the cut\_models.py script. You can call this script as shown below. It will produce a series of files, located in a folder called edge\_model\_<network name>.

`sudo python3 cut_models.py <network_name>`

The benchmarking software will load these models on the Edge to speed up benchmarking. This has already been done for the 8 models we used in our research.

Using the benchmarking software to conduct experiments should be straightforward. The server should be launched first and can be done so as shown below.

`sudo python3 benchmarking_server.py <network_name>`

The use of ‘sudo’ is important. The benchmarking software on the Edge artificially reduces the network speed with the built-in ‘tc’ command. Updating network rules for ‘tc’ requires ‘sudo’ and therefore the benchmarking software must also be called with ‘sudo’.

If you try to launch the server without providing the name of a network to benchmark, the following error will be displayed:

_Run like this: sudo python3 benchmarking\_server.py <network_name>_

Once the server has been launched, the Edge node can be launched. It’s important to run the server first because unless the server is running the client won’t have anything to connect to and will timeout.

`sudo python3 benchmarking_client.py <server_address> <network_name>`

The client expects two parameters to be provided, via the command line. One for the server address and one for the network name. Failing to provide the required parameters will result in an error being displayed like the one shown below:

_Run like this: sudo python3 benchmarking\_client.py <server\_address> <network\_name>_

Network connections can be unreliable, therefore the benchmarking software will detect failed measurements and automatically re-do them until it can record correct measurements. Therefore there is no need for internal error reporting for the aspect of the software.

