Climate Model
=============

Introduction
------------

From the introduction of :cite:`leach_fairv200_2021`.

Earth System Models (ESMs) are vital tools for providing insight into the drivers behind Earth’s climate system,
as well as projecting impacts of future emissions. Large scale multi-model studies, such as the Coupled Model 
Intercomparison Projects (CMIPs), have been used in many reports to produce projections of what the future climate may look 
like based on a range of different concentration scenarios, with associated 
emission scenarios and socioeconomic narratives quantified by Integrated Assessment Models (IAMs). In addition to simulating 
both the past and possible future climates, these CMIPs extensively use idealised experiments to try to determine some of 
the key properties of the climate system, such as the equilibrium climate sensitivity (ECS), or the 
transient climate response to cumulative carbon emissions.

While ESMs are integral to our current understanding of how the climate system responds to GHG and aerosol emissions, and 
provide the most comprehensive projections of what a future world might look like, they are so computationally expensive 
that only a limited set of experiments are able to be run during a CMIP. This constraint on the quantity of experiments 
necessitates the use of simpler models to provide probabilistic assessments and explore additional experiments and scenarios. 



These models, often referred to as simple climate models (SCMs), are typically designed to emulate the response of more complex models.
In general, they are able to simulate the globally averaged emission wich are convert into concentration then into radiative forcing and finally into temperature response pathway, 
and can be tuned to emulate an individual ESM (or multi-model-mean). 


In general, SCMs are considerably less complex than ESMs: while ESMs are three dimensional, 
gridded, and explicitly represent dynamical and physical processes, therefore outputting many hundreds of variables, 
SCMs tend to be globally averaged (or cover large regions), and parameterise many processes, resulting in many fewer output variables. 
This reduction in complexity means that SCMs are much quicker than ESMs in terms of runtime: most SCMs can run tens of thousands of 
years of simulation per minute on an “average” personal computer, whereas ESMs may take several hours to run a single year 
on hundreds of supercomputer processors; and are much smaller in terms of the number of lines of code: 
SCMs tend to be on the order of thousands of lines, ESMs can be up to a million lines.


.. bibliography::


Diagram
~~~~~~~

.. tikz:: 

	\begin{tikzpicture}
		\begin{pgfonlayer}{nodelayer}
			\node [style=none] (0) at (-6, -5.75) {};
			\node [style=none] (1) at (9.25, -5.75) {};
			\node [style=none] (2) at (-6, 9.75) {};
			\node [style=none] (3) at (9.25, 9.75) {};
			\node [style=none, label={below right : Temperature dynamic}] (4) at (-4.75, -1) {};
			\node [style=none] (5) at (8.75, -1) {};
			\node [style=none] (6) at (8.75, -5) {};
			\node [style=none] (7) at (-4.75, -5) {};
			\node [style=none, label={below right : Carbon Cycle}] (10) at (-4.75, 9.25) {};
			\node [style=none] (11) at (8.75, 9.25) {};
			\node [style=none] (12) at (8.75, 4.75) {};
			\node [style=none] (13) at (-4.75, 4.75) {};
			\node [style=none, label={below right : Radiative Forcing}] (14) at (-4.75, 3.75) {};
			\node [style=none] (15) at (8.75, 3.75) {};
			\node [style=none] (16) at (8.75, 0) {};
			\node [style=none] (17) at (-4.75, 0) {};
			\node [style=new style 0] (18) at (-2, 7.75) {$d_C$};
			\node [style=new style 1] (19) at (0, 7.75) {$+$};
			\node [style=new style 1] (20) at (-3.5, 7.75) {$+$};
			\node [style=none, label={below  :$E_{EX}$}] (21) at (-3.5, 5.75) {};
			\node [style=new style 0] (22) at (2.25, 7.75) {$z^{-1}$};
			\node [style=none] (23) at (-4.25, 2.25) {$C_{AT}$};
			\node [style=none] (24) at (4.25, 7.75) {};
			\node [style=none] (25) at (4.25, 6) {};
			\node [style=none] (26) at (0, 6) {};
			\node [style=new style 0] (27) at (6.75, 7.75) {$b_C$};
			\node [style=new style 0] (28) at (-2, -2.25) {$d_\Theta$};
			\node [style=none, label={right : $\Theta_{AT}$}] (33) at (10, -2.25) {};
			\node [style=new style 0] (37) at (6.75, -2.25) {$b_\Theta$};
			\node [style=none] (38) at (6.75, 4.25) {};
			\node [style=none] (39) at (-5.25, 4.25) {};
			\node [style=new style 1] (40) at (-2.5, 2) {$\times$};
			\node [style=new style 0] (41) at (0.25, 2) {$\log_2$};
			\node [style=new style 1] (42) at (2.75, 2) {$\times$};
			\node [style=none, label={below : $F_{CO2}$}] (43) at (2.75, 0.75) {};
			\node [style=none] (44) at (5.75, -0.5) {};
			\node [style=none] (45) at (-5.25, -0.5) {};
			\node [style=none, label={left : E}] (47) at (-7, 7.75) {};
			\node [style=none] (48) at (-5.25, 2) {};
			\node [style=new style 0] (50) at (2.25, 6) {$A_C$};
			\node [style=new style 1] (51) at (0.25, -2.25) {$+$};
			\node [style=new style 0] (52) at (2.5, -2.25) {$z^{-1}$};
			\node [style=none] (53) at (4.5, -2.25) {};
			\node [style=none] (54) at (4.5, -4) {};
			\node [style=none] (55) at (0.25, -4) {};
			\node [style=new style 0] (56) at (2.5, -4) {$A_\Theta$};
			\node [style=none, label={below: $1/C_{AT}(ref)$}] (57) at (-2.5, 0.75) {};
			\node [style=none] (58) at (-5.25, -2.25) {};
			\node [style=new style 1] (59) at (5.75, 2) {$+$};
			\node [style=none, label={right  :$F_{EX}$}] (60) at (7.25, 2) {};
		\end{pgfonlayer}
		\begin{pgfonlayer}{edgelayer}
			\draw [style=new edge style 1] (3.center)
				to (1.center)
				to (0.center)
				to (2.center)
				to cycle;
			\draw [style=new edge style 3] (4.center)
				to [in=180, out=0] (5.center)
				to (6.center)
				to (7.center)
				to cycle;
			\draw [style=new edge style 3] (10.center)
				to (11.center)
				to (12.center)
				to (13.center)
				to cycle;
			\draw [style=new edge style 3] (16.center)
				to (17.center)
				to (14.center)
				to [in=180, out=0] (15.center)
				to cycle;
			\draw [style=new edge style 0] (20) to (18);
			\draw [style=new edge style 0] (18) to (19);
			\draw [style=new edge style 0] (21.center) to (20);
			\draw [style=new edge style 0, label={above :$n$}] (19) to (22);
			\draw (22) to (24.center);
			\draw [style=new edge style 0] (24.center) to (25.center);
			\draw [style=new edge style 0] (25.center) to (50);
			\draw (50) to (26.center);
			\draw (26.center) to (19);
			\draw [style=new edge style 0] (24.center) to (27);
			\draw [style=new edge style 0] (37) to (33.center);
			\draw (27) to (38.center);
			\draw (38.center) to (39.center);
			\draw [style=new edge style 0] (40) to (41);
			\draw [style=new edge style 0] (43.center) to (42);
			\draw [style=new edge style 0] (41) to (42);
			\draw (44.center) to (45.center);
			\draw [style=new edge style 0] (47.center) to (20);
			\draw (39.center) to (48.center);
			\draw (48.center) to (40);
			\draw [style=new edge style 0, label={above :$n$}] (51) to (52);
			\draw (52) to (53.center);
			\draw [style=new edge style 0] (53.center) to (54.center);
			\draw [style=new edge style 0] (54.center) to (56);
			\draw (56) to (55.center);
			\draw (55.center) to (51);
			\draw [style=new edge style 0] (53.center) to (37);
			\draw [style=new edge style 0] (28) to (51);
			\draw [style=new edge style 0] (57.center) to (40);
			\draw (45.center) to (58.center);
			\draw [style=new edge style 0] (60.center) to (59);
			\draw [style=new edge style 0] (42) to (59);
			\draw (59) to (44.center);
			\draw [style=new edge style 0] (58.center) to (28);
		\end{pgfonlayer}
	\end{tikzpicture}

   :libs: arrows

Here the diagram of the SCM structure we will use.

Constants and Parameters
------------------------

Constants
~~~~~~~~~

.. automodule:: models.geophysic_models.constants
   :members:

Parameters
~~~~~~~~~~

.. automodule:: models.geophysic_models.SCM_parameters
   :members:


Carbon Cycle Models
-------------------

.. automodule:: models.geophysic_models.carbon_cycle_models
   :members:


Radiative Forcing
-----------------

.. automodule:: models.geophysic_models.radiative_forcing
   :members:

Temperature Dynamic Models
--------------------------

.. automodule:: models.geophysic_models.temperature_dynamic_model
   :members:

Simple Climate Model
--------------------
   
.. automodule:: models.geophysic_models.climate_model
   :members: