import os
import os.path

sufixsrc = '_professor.ipynb'
sufixtg = '_student.ipynb'

for dirpath, dirnames, filenames in os.walk("."):
    for filename in [f for f in filenames if f.endswith(sufixsrc)]:
        srcfile = os.path.join(dirpath, filename)
        tgfile = srcfile.replace(sufixsrc, sufixtg)
        print('Processing file:', srcfile)
        with open(srcfile, 'r') as fin:
        	with open(tgfile, 'w') as fout:
        		state = True
        		skip_next = False
        		for line in fin:
        			if state:
        				if '<FILL IN>' in line:
        					skip_next = True
        					fout.write(line)
        				else:
        					if skip_next:
        						skip_next = False
        					else:
        						fout.write(line)
        				if '<SOL>' in line:
        					state = False
        			else:
        				if '</SOL>' in line:
        					fout.write('\n'+line)
        					state = True

        os.system('jupyter nbconvert --to html ' + tgfile.replace(' ', '\ '))
        os.system('jupyter nbconvert --to html ' + srcfile.replace(' ', '\ '))



