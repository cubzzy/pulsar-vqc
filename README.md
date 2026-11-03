This branch contains extra code that's still relevant but not important to the main pipeline.

**plot_runs.py**
- Contains a script that parses through the .csv files in plot_data, then plots all of the loss curves on one graph and all the ROC curves on another
- We obtained the loss and ROC data in plot_data by modifying our report_maker.py script to output this data to a .csv, and we plotted each config's best seed from seeds 1-5. This modification wasn't committed; it was run offline on one person's machine.
- We didn't include TwoLocal Full Entanglement because it has the same values of TwoLocal with Circular Entanglement, so it would be redundant
