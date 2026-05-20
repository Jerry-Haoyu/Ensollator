# Ensollator 

<!-- ## Project Description 
### The Climate Context
The El Niño Southern Oscillation (ENSO) is a unique and prominent intrinsic variability (i.e., free of external forcing, unlike seasonal variations) of our planet. It dictates a massive portion of the global climate system and has significant socioeconomic influences. It primarily oscillates between three states:

- **Neutral**: Global temperatures remain near long-term averages. The western Pacific (near northern Australia and Indonesia) is warm and rainy, while the eastern Pacific (near South America) stays cool and dry.

- **El Niño**: A general warming of global atmospheric temperatures. The central and eastern tropical Pacific become significantly hotter than usual, shifting heavy rainfall eastward. This often causes hotter, drier conditions in Australia, Southeast Asia, and parts of northern Brazil, while bringing unusually wet weather to the southern United States and coastal South America.

- **La Niña**: A period of lower than normal global atmospheric temperatures. The central and eastern tropical Pacific become much colder than usual. This setup often intensifies drought in the southern United States and coastal Peru, while bringing cooler, wetter conditions to the Pacific Northwest, northern Australia, and Southeast Asia.

### Thermocline Depth as an ENSO Indicator

To quantitatively model ENSO, a commonly used climate index is Nino 3.0, which is a spatial average of sea surface temperature (SST) on a rectangular domain in the east equatorial Pacific region. In ideal models, SST is somewhat implicit. An alternative metric called thermocline depth is often used instead.The thermocline is the well-mixed, warm layer of sea near the surface, sitting above the cold, salty deep sea. The thermocline depth $h$ is closely related to SST because a deeper thermocline means a larger surface layer of the sea is warm and well-mixed. This relationship can be expressed as:$$h \sim \Theta(T_{\text{surf}})$$We use a spatial average $h_E$ of $h$ (similar to Nino 3.0) as our indicator. Suppose $\Omega$ is an open rectangle of the equatorial Pacific that we are modeling, and let $S \subset \Omega$ be the eastern half of that rectangle:$$h_E = \frac{1}{|S|}\int_{S}h_E(\mathbf{x})$$

### Delayed Oscillator

The delayed oscillator model describes the idealistic oscillation behavior of the spatial mean of an ENSO-characterizing climate variable $h$. This variable indicates the current phase of ENSO (examples include east-basin-mean SST or east-basin-mean thermocline depth anomaly). The model is a straightforward system of Ordinary Differential Equations (ODEs):$$\frac{dh}{dt} = ah - bh(t-\delta) - rh^3$$Note: This system is a non-linear, first-order, infinite-dimensional system since we essentially have $|(0,\delta)| = 2^{\aleph_0}$ equations.

### Coupled Shallow-Water System

![]()

A slightly more complex idealistic model of ENSO is the coupled shallow-water system. The main climate variables are captured by the functions $\mathbf{u}$, $\mathbf{U}$, $h(x,y,t)$, and $\phi(x,y,t)$, where:$\mathbf{u} = (u,v)$ is the oceanic velocity.$\mathbf{U} = (U,V)$ is the atmospheric counterpart.$h$ is the thermocline depth perturbation (indicating the ENSO phase).$\phi$ is the atmospheric pressure.$$\partial_t \mathbf{u} = -f\mathbf{k} \times \mathbf{u}-g'h -r\mathbf{U}$$

$$\partial_t h = -H\nabla \cdot h$$
$$\partial_t \mathbf{U} = -f\mathbf{k} \times \mathbf{U}-g'\phi$$

$$\partial_t \phi = \alpha h-c_a^2\nabla \cdot \phi$$

Here, $c_a$ is the atmospheric gravity wave speed and $H$ is the mean thermocline depth perturbation.The ocean-air interaction comes into play via two linear terms:$-r\mathbf{U}$ acts as the wind stress forcing (air to sea).$\alpha h$ acts as the thermodynamic forcing (sea to air).The shallow water model has 2D rectangular domains $\Omega_o$ and $\Omega_a$. This 3D-to-2D reduction allows for a simple finite difference method simulation as opposed to more complex finite volume methods.

### The Learning Task
The Mathematical StructureLet $\Omega = (a,b) \times (c,d)$ be an open rectangle. Consider a translational-invariant boundary value problem (BVP)$$\mathcal Lu(\mathbf{x},t) = f \quad \mathbf{x} \in \Omega, t \in (0,\infty)$$
$$\mathcal Bu(\mathbf{x},t) = g \quad \mathbf{x} \in \partial \Omega, t \in (0,\infty)$$

$$u(\mathbf{x},t_0) = u_0 \quad \mathbf{x} \in \Omega$$

> **Definition**: (S-moment) Let the boundary problem be defined as above. Let $S \subset \Omega$ be an open connected subset. Define the S-moment as:$$u_S(t) = \int_{\mathbf{x}\in S} u(\mathbf{x},t)$$

On the other hand, let:

$$\mathcal Ny(t) = g \quad t \in (0,\infty)$$

$$y(t_0) = y_0$$

be a one-dimensional non-linear system, where the linear operator $\mathcal N$ is characterized by a sequence of finite parameters $(a,b,c,\dots)$. We will assume there are 4 parameters $(a,b,c,d)$ from now on to adhere to our specific ENSO problem.In addition, let $\mathcal Y = \mathcal Y((0,\infty);\mathbb{R})$ be a set of separable Banach Space. Let $G: \mathbb{R}^4 \to \mathcal Y((0,\infty);\mathbb{R})$ be the "solution map" that, given $(a,b,c,d)$, produces the solution $y \in \mathcal Y((0,\infty);\mathbb{R})$. 

#### Best S-moment Approximator
> **Definition**: (Best 
$u_S(t)$-approximator) Given a BVP defined as above, define the best $u_S(t)$-approximator as $G(\widehat \Theta) \in \mathcal Y((0,\infty);\mathbb{R})$ where the parameters $\widehat \Theta = (a,b,c,d)$ are:$$\widehat \Theta = \mathrm{argmin}_{\Theta} \int_t \left(u_S(t)-G(\Theta)\right)^2$$

This task can easily be formulated as a non-linear least-squares problem. In the case of the delayed oscillator, we have empirical pairs $\left(t, \frac{dh}{dt}\right)$ and:$$f(t,[a,b, r,\delta]) = ah(t) - bh(t-\delta) - rh^3$$where $h$ is passed in as a global variable (e.g., an array with $O(1)$ look-up) and $h(t-\delta)$ is given by interpolation.Given a family of BVPs instantiated with different parameters, it is highly desirable to find a map between those parameters and the best $S$-moment approximator. In our ENSO problem, this establishes a robust connection between our two most important idealistic models.

 > **Definition**: (BVP Parameter Map) Denote a family of BVPs, each specified by a family of parameters $\{\Pi(\boldsymbol\xi)\}, \boldsymbol{\xi} \in \mathbb{R}^k$. Let $F: \mathbb{R}^k \to \mathbb{R}^4$ be the map that connects the BVP $\Pi$ to the oscillation parameters $\tilde \Theta$ of the best $S$-moment approximator.

An example of parameters $\boldsymbol{\xi}$ characterizing the BVP in the ENSO shallow-water model might be $(r, \alpha)$—the strength of the air-sea coupling. However, this mapping can go further to parametrize initial conditions, boundary conditions, and external forcings.Studying $F$ is incredibly valuable because many parameters in the shallow-water model do not have a clear, direct translation to overall oscillatory behavior. For instance, the relationship of basin width to the strength of positive feedback, negative feedback, and damping is typically quite opaque.More generally, finding a direct functional relationship that maps the parameters of the shallow-water model ($\boldsymbol{\xi}$) to the oscillation parameters ($\Theta$) translates the state of a higher-dimensional climate system into a simple, robust, and concise description of oscillatory behavior. -->


# Repo Structure 
The source direcotry has the following structure:
```
src
├── oscillatorNet
│   ├── data_loader.py
│   ├── oscillatorNet.py
│   └── train.py
├── oscillator_data_generation
│   ├── __init__.py
│   ├── delayed_oscillator_solver.py
│   └── generate_data.py
└── shallow_water_solver
    ├── config.jl
    ├── fft.jl
    ├── shallow_water.jl
    └── visualize.jl
```

## Data Generation
The `oscillatorNet` directory contains the training pipeline for OscNet, the  `oscillattory_data_generation` generaets all the data with the custom heun solver in the `delayed_oscillator_solver.py` script and the grid search in `generate_data.py` which leverages multiprocessing. It is recommended to use multicore CPU inorder to generate a large training data set. To modify the size of the dataset, i.e., the sampling density, modify the following variables in  `generate_data.py`:

```python
 #### Sampling Density #####
    N_deltas = 5
    N_as = 5
    N_bs = 5
    N_rs = 5
###########################
```
For a good training data set, use ~30 for each. A makefile is prepared to launch training direcly, after setting the parameters in the script, start simulation by:

```Makefile
make oscNetData
```

## Training 
THe pipeline consits of three classes 
```python
class oscNetData(Dataset)
class OscNet(nn.Module)
class trainer()
```
The first class and the second class is a conventional torch set up, the third class is a custom class that automates directory creation, custom real-time loss plot generation(in the output folder). Start training by:

```
make trainOscNet:
```