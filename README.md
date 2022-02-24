# Pyramid Anomaly Detection

Salut mes amis et je souhaite
vous etes chakhmakakh! Alors, this repository contains raw implementation of 
the following anomaly detection algorithms:

* Distance to All: Distance of the point to all points is calculated, and aggregated.
* Distance to Nearest Neighbor: Distance of the point to the nearest neighbor is calculated, and aggregated.
* Distance to K-Nearest Neighbors: Distance to K-Nearest neighbors is calculated, and aggregated. With this one, you can either use a named aggfunc (mean, sum, median) or define it yourself through a Lambda expression and it will be evaluated in runtime.
* K-Means Clustering: The usual K-Means deal. Generates k clusters of points, k is given. The output for the previous three is a CSV file, but for this one, it's a JSON file.
* Fuzzy C-Means Clusteinr: The fuzzy version of K-Means, this time ,the probability is calculated. The output is a chart, and not a dataset.


How to run:

1. Install Python 3.10 (and only 3.10!)

2. Create a virtual env with `python3.10 -m virtualenv [envname]`

3. `source [envname]/bin/activate` (On Linux)

Now you have two ways to run the application.

**A)** Through commandline

Just run `python3 ad.py -h` to see the help pop up, then decide the parameters. I don't see the need to explain 
every thing here because the help explains everything clearly.

**B)** Edit `params.json` and run `python3 adjson.py` --- keep in mind that you can give it path to another 
params.json like `python3 adjson ./params2.json`.
                    
Inside `params.json` there's a parameter called `python_executable`. If you are using
virtualenv, and you have currently activated it, set it to `python3`. Otherwise, give it the path to your Python 3.10 installation!

If you wish to use a default value, use `DEFAULT` or `default`.

Any questions, direct it at Chubak#7400 on Discord.
                                     
You can use the examples file to test things out.

> Distance functions are from SciPy but the main algorithms are by me using np.vectorize, fastest possible!


Et alors je bid a revioir!
 