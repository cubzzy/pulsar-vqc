This branch contains extra code that's still relevant but not important to the main pipeline.

**plot_runs.py**
- Contains a script that parses through the .csv files in plot_data, then plots all of the loss curves on one graph and all the ROC curves on another
- We obtained the loss and ROC data in plot_data by modifying our report_maker.py script to output this data to a .csv, and we plotted each config's best seed from seeds 1-5. This modification wasn't committed; it was run offline on one person's machine.
- We didn't include TwoLocal Full Entanglement because it has the same values of TwoLocal with Circular Entanglement, so it would be redundant

**ionq_inference.py**
- Runs the trained weights from ZZ + EfficientSU2 + Full Entanglement 
- Outputs a .csv file containing the probabilities for each pulsar in the testing set (of which there are 300)

**compare_backends.py**
- Compares the .csv probability files generated from ionq_inference.py
- Outputs a scatter plot showing the comparison between the predictions for each object in the test set
- Prints to the terminal information about the number of label flips (about something being a pulsar or not), the confidence slope. For example, if the confidence slope is 0.880, then the noisy model is 12% less confident
- Prints to the terminal the mean p1 (prediction of something being a pulsar vs not) of a non pulsar and a pulsar for both aer and IonQ, as well as the AUCs for each.