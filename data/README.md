# Feature Description

## General Information
- *Network*, *Configuration*, *Run*, *Repetition* are irrelevant for the prediction and are merely identifiers for each simulation

## Dynamic Performance Metrics
- The metrics we want to predict, mean and maximum value
	- *RTT*: Round-trip time of pings
	- *ControlPlaneTraffic*: S2C traffic caused by e.g. mismatched packets or topology discovery
	- *SyncTraffic*: C2C traffic due to synchronization, for hierarchical architecture (here: Kandoo) only with respect to the local controllers

## Semi-Dynamic Controller Metrics
- These metrics may change per configuration or timeout
	- *TimeOut*: Idle timeout of flow entries
	- *Controllers*: Number of controller instances
	- *SwitchesPerController*: Switches connected to a specific instance, extracted as absolute and relative metric for each controller, but also statistics (mean, min, max, median, mode, std, var, varcoeff, skew, kurt) over all controllers
	- *SwitchMappingEntropy*: Entropy of switch mappings as a measure of mapping-balance
	- Various delays (+ various statistical measures each [mean, min, max, median, mode, std, var, varcoeff, skew, kurt])
		- Include the processing delay at switches and the propagation delay inbetween them
		- *C2SL*: Controller-to-Switch Delay
		- *C2CL*: Controller-to-Controller Delay
		- *C2RL*: Controller-to-Root Delay (only meaningful for Kandoo, is dropped during training for HyperFlow)

## Static Topology Metrics
- These metrics are always the same for a network and persist through different configurations
- For full mathemetical explanations for some of the more complex metrics, see the paper *"Classification of graph metrics"* by Hernández and Van Mieghem, also cited in our paper in the dataset section
	- Local/node-based metrics (+ various statistical measures each [mean, min, max, median, mode, std, var, varcoeff, skew, kurt])
		- *HopCount*: Statistics over shortest path lengths
		- *E2EL*: Basically weighted *HopCount* with latency, End-to-End delay of all paths between all switches on a path, includes the processing delay at switches and the propagation delay inbetween them
		- *NodeEccentricity*: Statistics of *longest* shortest path to other switches for each switch, weighted and unweighted
		- *Farness*: Statistics over average shortest paths, weigthed and unweighted + normalized and unnormalized
		- *Closeness*: Reprocical of *Farness*, weigthed and unweighted + normalized and unnormalized
		- *Betweenness*: Statistics over fraction of shortest paths a node is contained in, weigthed and unweighted + normalized and unnormalized
		- *Degree*: Self-explanatory, normalized and unnormalized
		- *S2SL*: basically weighted version of degree, average latency to *neighbouring* switches
		- *EdgeDistinctPaths*: Statistics over edge-distinct paths for all node-pairs
		- *NodeDisjointPaths*: Statistics over node-disjoint paths for all node-pairs
		- *LocalClusteringCoefficient*: Statistics over degree of meshing between neighbours of a node
		- *NodeExpansion*: Calculates the fraction of nodes within a fixed ballradius, for four different values dependent on (unweighted) *Diameter*, last expansion should always be 1
	- Global/graph-based metrics (single value each)
		- *GraphEccentricity*: Average *NodeEccentricity*, weighted and unweighted
		- *Radius*: Minimum *NodeEccentricity*, weighted and unweighted
		- *Diameter*: Maximum *NodeEccentricity*, weighted and unweighted
		- *GraphCentralization*: Measure of maximum degree/centralization of a graph (_not_ equivalent to maximum *Degree*) 
		- *Assortativity*: Measure  of if high-degree nodes attach to other high-degree nodes or low-degree nodes
		- *CentralPointDominance*: Measure of maximum betweenness/centralization of a graph (_not_ equivalent to maximum *Betweenness*)
		- *VertexConnectivity*: Equivalent to maximum *NodeDisjointPaths*
		- *VertexPersistence*:  Equivalent to maximum *NodeDisjointPaths* >= *Diameter*
		- *EdgeConnectivity*: Equivalent to maximum *EdgeDistinctPaths* 
		- *EdgeConnectivity*: Equivalent to maximum *EdgeDistinctPaths*  >= *Diameter* 
		- *GlobalClusteringCoefficient*: Equivalent to average *LocalClusteringCoefficient*
		- *GraphExpansion*: Equivalent to average *NodeExpansion* for all four values
		- *RichClubCoefficient*: For four values dependent on number of nodes contained in the rich-club, a rich-club only contains nodes of a certain degree or higher

