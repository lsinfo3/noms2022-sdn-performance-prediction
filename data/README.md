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
	- *SwitchesPerController_CX*: Raw number of switches connected to controller instance *X* with *X* in {1,2,3,4,5}
	- *SwitchesPerControllerNormalized_CX*: Normalized *SwitchesPerController_CX* by dividing by |*Switches*|
	- *SwitchesPerController*: Statistics (mean, min, max, median, mode, std, var, varcoeff, skew, kurt) of switches connected to a controller instance over all controllers
	- *SwitchesPerControllerNormalized*: Normalized *SwitchesPerController* by dividing by |*Switches*|
	- *SwitchMappingEntropy*: Shannon entropy of switch mappings as a measure of mapping-balance, e.g., for two *Controllers*: -(p1 * log2(p1) + p2 * log2(p2)), where p1 is the relative frequency of *Switches* connected to *Controller* 1 and p2 is the relative frequency of *Switches* connected to *Controller* 2
	- *SwitchMappingEntropyNormalized*: Normalized *SwitchMappingEntropy* by dividing by log2(*|Controllers*|) (maximum Shannon entropy)
	- Various delays (various statistical measures each [mean, min, max, median, mode, std, var, varcoeff, skew, kurt])
		- Include the processing delay at switches and the propagation delay inbetween them
		- *C2SL*: Controller-to-Switch Latency
		- *C2CL*: Controller-to-Controller Latency
		- *C2RL*: Controller-to-Root Latency (only meaningful for Kandoo, is dropped during training for HyperFlow)

## Static Topology Metrics
- These metrics are always the same for a network and persist through different configurations
- **Note**: For full mathemetical explanations for some of the more complex metrics, see the paper *Classification of graph metrics* by Hernández and Van Mieghem, also cited in our paper in the dataset section, as the implementations in the OOS adhere to their definitions, as stated in the paper *Simulative Evaluation of KPIs in SDN for Topology Classification and Performance Prediction Models* by Gray et al., which we further extended
	- Local/node-based metrics (+ various statistical measures each [mean, min, max, median, mode, std, var, varcoeff, skew, kurt])
		- *HopCount*: Shortest path lengths
		- *E2EL*: End-to-End Latency of all paths between all switches on a path, includes the processing delay at switches and the propagation delay inbetween them, basically weighted *HopCount* with latency
		- *NodeEccentricity*: *Longest* shortest paths to other switches for each switch
		- *NodeEccentricity_weighted*: *NodeEccentricity* weighted with latency
		- *Farness*: Shortest paths per switch (sum of all shortest path lengths to other switches)
		- *FarnessCentrality*: Normalized *Farness* by dividing by (|*Switches*| - 1) (basically average path length to other switches then) (Sidenote: this is slighty differently aggregated then the *HopCount*, here is first aggregated over a single switch, than over all switches, *HopCount* is aggregated directly over all paths)
		- *Farness_weighted*: *Farness* weighted with latency
		- *FarnessCentrality_weighted*: *FarnessCentrality* weighted with latency
		- *Closeness*: Reprocical of *Farness*
		- *ClosenessCentrality*: Normalized *Closeness* by multiplying with (|*Switches*| - 1) 
		- *Closeness_weighted*: *Closeness* weighted with latency
		- *ClosenessCentrality_weighted*: *ClosenessCentrality* weighted with latency
		- *Betweenness*: Fraction of shortest paths over all node-pairs a switch is contained in
		- *BetweennessCentrality*: Normalized *Betweenness* by dividing by (|*Switches*| - 1) * (|*Switches*| - 2)
		- *Betweenness_weighted*: *Betweeness* calculated with respect to shortest paths weighted with latecy
		- *BetweennessCentrality_weighted*: *BetweennessCentrality* calculated with respect to shortest paths weighted with latecy
		- *Degree*: Degree of switches (here in = out-degree, so no distinction)
		- *DegreeCentrality*: Normalized *Degree* by dividing by (|*Switches*| - 1)
		- *S2SL*: Switch-to-Switch Latency/(single-)link latency (i.e., directly aggregated over all link latencies, not first for each switch)
		- *EdgeDistinctPaths*: Edge-distinct paths for all node-pairs
		- *NodeDisjointPaths*: Node-disjoint paths for all node-pairs
		- *LocalClusteringCoefficient*: Degree of meshing between neighbours of a node
		- *NodeExpansion*: Calculates the fraction of nodes within a fixed ballradius, for four different values dependent on (unweighted) *Diameter* (25%, 50%, 75%, and 100% of *Diameter*), last expansion should always be 1
	- Global/graph-based metrics (single value each)
		- *Hosts*: Number of end-devices
		- *Switches*: Number of switches (here: *Switches* = *Hosts*/2)
		- *Links*: Number of connections between switches
		- *GraphEccentricity*: Average *NodeEccentricity*
		- *GraphEccentricity_weighted*: Average *NodeEccentricity_weighted*
		- *Radius*: Minimum *NodeEccentricity*
		- *Radius_weighted*: Minimum *NodeEccentricity_weighted*
		- *Diameter*: Maximum *NodeEccentricity*
		- *Diameter_weighted*: Maximum *NodeEccentricity_weighted*
		- *GraphCentralization*: Measure of maximum degree/centralization of a graph (_not_ equivalent to maximum *Degree*) 
		- *CPD*: Central Point Dominance, measure of maximum *Betweenness* of the newtork (_not_ equivalent to maximum *Betweenness*)
		- *CPD_weighted*: Central Point Dominance, measure of maximum *Betweenness_weighted* of the newtork (_not_ equivalent to maximum *Betweenness_weighted*)
		- *Assortativity*: Measures if high-degree nodes attach to other high-degree nodes or low-degree nodes*)
		- *VertexConnectivity*: Equivalent to maximum *NodeDisjointPaths*
		- *VertexPersistence*:  Equivalent to maximum *NodeDisjointPaths* >= *Diameter*
		- *EdgeConnectivity*: Equivalent to maximum *EdgeDistinctPaths* 
		- *EdgePersistence*: Equivalent to maximum *EdgeDistinctPaths*  >= *Diameter* 
		- *GlobalClusteringCoefficient*: Equivalent to average *LocalClusteringCoefficient*
		- *GraphExpansion*: Equivalent to average *NodeExpansion* for all four values
		- *RichClubCoefficient*: A rich-club only contains nodes of a certain degree or higher, for four values dependent on number of nodes contained in the rich-club (25%, 50%, 75%, and 100% of nodes)

