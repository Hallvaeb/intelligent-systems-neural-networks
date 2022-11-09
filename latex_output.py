def latex_out(results, no_of_targets):
	bs = '\\'

	lat_out = open('out_latex.txt', 'w')
	lat_out.write(bs+'begin{figure}\n'+bs+'raggedright\n')
	lat_out.close()

	for i in range(no_of_targets):
		confusion, acc = results[i][3], results[i][2]	
		output = open('out_latex.txt', 'a')
		# Convert to strings to put right allow concatination with rest of string
		tp = confusion[0][0]
		fp = confusion[0][1]
		fn = confusion[1][0]
		tn = confusion[1][1]
		t = tp+fp
		tp_fp = f'{t}'
		fn_tn = f'{fn+tn}'
		tp_fn = f'{tp+fn}'
		fp_tn = f'{fp+tn}'
		n = f'{tp+fp+fn+tn}'

		acc = f'{acc}' 
		trial_no_str = f'{i+1}' 
		output.write(
			bs+'begin{subfigure}{.45'+bs+'textwidth}\n'+bs+'begin{subfigure}{.9'+bs+'textwidth}\n'+bs+'begin{tabular}{l|l|c|c|c}\n'+
			bs+'multicolumn{2}{c}{}&'+bs+'multicolumn{2}{c}{Actual}&'+bs+bs+'\n'+bs+'cline{3-4}\n'+
			bs+'multicolumn{2}{c|}{}&Positive & Negative &'+
			bs+'multicolumn{1}{c}{Total}'+bs+bs+'\n'+
			bs+'cline{2-4}\n'+
			bs+'multirow{2}{*}{'+bs+'rotatebox[origin=c]{90}{'+bs+'parbox[c]{1.3cm}{'+bs+'centering Predicted}}}&Positive &'+f'{tp}'+' & '+f'{fp}'+' & {'+tp_fp+'}'+bs+bs+'\n'+
			bs+'cline{2-4} & Negative & '+f'{fn}'+' & '+f'{tn}'+' & {'+fn_tn+'}'+bs+bs+'\n'+
			bs+'cline{2-4}\n'+
			bs+'multicolumn{1}{c}{} & '+bs+'multicolumn{1}{c}{Total} & '+bs+'multicolumn{1}{c}{'+tp_fn+'}&'+
			bs+'multicolumn{1}{c}{'+fp_tn+'} & '+bs+'multicolumn{1}{c}{'+n+'}\n'+
			bs+'end{tabular}\n'+
			bs+'end{subfigure}\n'+bs+'centering\nAccuracy: '+acc+bs+bs+'\n'+
			bs+'label{fig:trials_conf_acc_'+trial_no_str+'}\n'+
			bs+'caption{Trial '+trial_no_str+'}\n'+
			bs+'end{subfigure}\n')
		print(f"Trial {i} appended to 'latex_out.txt'.")
	output.write(
		bs+'caption{Confusion matrices and accuracies}\n'+
		bs+'label{fig:trials_conf_acc}\n'+
		bs+"end{figure}")
	print("All trials written to 'latex_out.txt'.")
