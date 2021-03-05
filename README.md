# AIOP_ML

This code makes an Imitation Controller / Dependant Variable by fitting all columns of data in series of 2d dataframes to the specified PV

### Data PreProcessor

This takes the output from the simulator and turns it into a 2d dataframe. Dataframes are saves as 'train1.csv',...,'train*.csv' in the working directory.  2d dataframes are helpful for plotting other analysis. The EnvMaker and SimMaker below also take the 2d dataframes as input.

valfile = is simulation data that should not be passed to the environment or to the simulator.  It is reserved for validing the environment and the policy.

### Define Variables

PV: (string)Process variable tag name that we want to learn to control.

### Controller Function
Controller Function is a fully connected dense NN model that fits data passed to the PV, (process variable). It "learns" the impact of the variables on the the PV. It requires modelname, file names of data, the PV and env_lookback.

modelname: file name to store the environment model ".h5" and supporting ".txt". these files then get called by the Simulator function.

trainfiles: 2d dataframes from the Data PreProcessor. Non-sequential data should be stored and passed in seperate files. Else this creates an abnormal system dynamic where the non sequeitial data intersects and throws off the model. Note: columns of data in seperate files should be aligned the same and have the same header names. If they are different, they will be rejected. (The Data PreProcessor takes care of this)

lookback: the history of data needed to capture the system dynamics. if there is a delay in the PV response to a MV change, the lookback should be long enough to catch it.  The default is 20s

Default: set to True to use the default paramaters to train and save the model. Else set to False and call Controller_Name.TrainCont(fc1_dims=32,fc2_dims=32,lr=.0001,ep=1000)

