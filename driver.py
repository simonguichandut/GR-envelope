import sys
import IO

if IO.load_params()['FLD']:
    from env_GR_FLD import MakeEnvelope
else:
    from env_GR import MakeEnvelope

def driver_envelope(Rphotkm):

    IO.make_directories()

    Verbose = True

    if Rphotkm == 'redo':
        Rphotkm = IO.get_phot_list()[::-1]
        Verbose = False
    
    problems,success = [],[]

    for R in Rphotkm:
        print('\n*** Calculating envelope with photosphere at %f km***\n'%R)
        try:
            env = MakeEnvelope(R,Verbose=Verbose)
            success.append(R)
            IO.write_to_file(R,env)
            print('Rphot=%s done'%str(R))
        except:
            problems.append(R)
            print('PROBLEM WITH Rphot = %.3f km'%R)
        
    print('\n\n*********************  SUMMARY *********************')
    print('Found solutions for these values : ',success)
    if len(problems)>=1:
        print('There were problems for these values : ',problems,'\n')


# Command line call
if __name__ == "__main__":
            
    i = sys.argv[1]
    
    if len(i)>10 and ',' not in i:
        sys.exit('Give Rphots separated by commas and no spaces')

    if i=='all':
        Rphotkm='all'
    elif i=='redo':
        Rphotkm='redo'
    else:
        if ',' in i:
            Rphotkm = [eval(x) for x in i.split(',')]
        else:
            Rphotkm = [eval(i),]

    driver_envelope(Rphotkm)

        
        
        
        
        
        

