I want to plot target alignment as a function of inference steps. 
I want to do this with a new rescaled learning rate also. 

To do this, use similar code to AE you have, and instead modify it to be an MLP. Then, instead of loss, plot target alignment after one weight update with 1, 5, 25, 125 etc inference steps.

Have the same approach as in figure 4: i.e. use batch learning, synthetic data etc but only do 1 learning step. Vary the number of inference steps instead. 

To do: 

Now you have things implemented