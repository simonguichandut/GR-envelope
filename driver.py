import sys
import IO

if IO.load_params()['FLD']:
    from env_GR_FLD import MakeEnvelope
else:
    from env_GR import MakeEnvelope

def driver_envelope(Rphotkm):

    from IO import write_to_file

    if Rphotkm == 'all': # a preset list of photospheric radii
        Rphotkm = (13,15,20,30,40,50,70,100,150,200,500,1000)

    problems,success = [],[]

    for R in Rphotkm:
        print('\n*** Calculating envelope with photosphere at %f km***\n'%R)
        try:
            env = MakeEnvelope(R)
            success.append(R)
            write_to_file(R,env)
        except:
            problems.append(R)
            print('PROBLEM WITH Rphot = %d km'%R)
        
    print('\n\n*********************  SUMMARY *********************')
    print('Found solutions for these values : ',success)
    if len(problems)>=1:
        print('There were problems for these values : ',problems,'\n')





# Command line call
if len(sys.argv)>1:
            
    i = sys.argv[1]
    
    if len(i)>4 and ',' not in i:
        sys.exit('Give Rphots separated by commas and no spaces')

    if i=='all':
        Rphotkm='all'
    else:
        if ',' in i:
            Rphotkm = [eval(x) for x in i.split(',')]
        else:
            Rphotkm = [eval(i)]

    driver_envelope(Rphotkm)

        
        
        
        
        
        

