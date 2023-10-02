1. For each source, we perform a local minimization and find a triplet in our parameter space (position and amplitude for each lens) - doing this for each source generates a list of candidate lenses that we can check.
	1. Where should we start our local minimization? [[Minimization start]]
2. Eliminate unrealistic lenses from our candidate list - this includes lenses with negative amplitude, lenses that are too large to be found, or too small to reasonably detect, lenses that fall too close to sources, etc. 
	1. What [[rejection criteria]] do we use?
3. Perform an iterative elimination to minimize the chi squared value of the candidate list - iteratively removing items from our remaining candidates to see if the chi squared improves or not. If it does, the lens stays removed - if it doesn't we restore the lens to our list. Once no lenses can be removed, we are finished.

# Next Steps

- Right now the local minimization uses a random starting point - I'm adding a simple calculation to choose a more reasonable starting point today. 
- Sometimes the 'iterative elimination' process eliminates _every_ lens - the implication being that the best fit to the sources is that all of the lensing comes from random noise. This is fine in principle - we want this pipeline to be able to conclude that no lensing has occurred - but right now I'm seeing it occur fairly often in cases where there definitely _is_ lensing. 
- I want to add a step with the iterative elimination to move our candidates around in the parameter space as well, to make sure that each is at the bottom of its local well. Down the road this might also involve splitting a given lens into _two_ lenses, each with half the amplitude, to see if the data is better fit by two lenses than one. This will be an important step of seeing if we can resolve small lenses that are close together.
- 

