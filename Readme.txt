Things to do first:
1. Start docker desktop(just open docker desktop, it will automatically start, check in left side)
2. extract this project, open in vs code

Things to setup in terminal:
1. Start minikube
	'minikube start'
2. Create Python's Virtual environment, if you not done before (if you see 'venv' folder, you are all setup!!)
  'python -m venv venv'
3. Activate venv
  './venv/Scripts/activate'  --> you should see (venv) at beginning, you are all setup!!
4. Install all packages, if you not done before.
  'python install -r requirements.txt'
5. Run the Python app
  'cd .\api\'
  'python .\paas_controller.py'  --> access the UI page through http://localhost:5000
6. Give the inputs in 'Phase-I'
  For Example:
  Application Name : my-web-app
  Operating System : Linux
  Architecture : 64-bit
  Base Image Size (GB) : 1.5
  Fill all the fields and click 'Deploy Application' button  --> This will print output below
7. Check the Deployments and Cronjobs
  'kubectl get pods'  --> list all the deployed pods with appropriate replicas
  'kubectl get cronjobs'  --> list all the cronjobs


What we are doing above?
  -> We first start a docker desktop which will setup a docker environment, then we start a minikube which will setup a kubernets environment. (Note one thing, To start minikube, docker should be in running condition)
  -> After all environment is done, we get into a python's virtual environment, and install all the required packages to run our project.
  -> Start our project which gives us UI using flask, in that we give all our inputs, and after deploy, which will given those input parameters to the AI and ML models and it predicted the CPU and RAM needs.
  -> It automatically creates deployment_yaml and cronjob_yaml and automatically applies those to the kubernets.
  -> By applying deployment in kubernets, it creates a pod which takes a dummy python's_server as an image and make container with it with a sizeof_image we mentioned in inputs.
  -> By applying cronjob in kubernets, it creates a job which act as a warmup program, which will run on every 1 hour (0 * * * *)
      the format is (minute hour day month week)
          (0 * * * *) --> every 0th minute like 1:00 2:00 3:00
          (0 * 1 * *) --> every 0th minute in 1st day of every month like 1st Jan 1:00,2:00,...23:00,24:00,1st Feb 1:00,2:00,...23:00,24:00,.....1st Dec 1:00,2:00,...23:00,24:00

That's All!!