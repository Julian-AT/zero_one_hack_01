"""NSPE — Neurosymbolic Process Engine.

Symbolic-first approach to the Infineon process-logic track. The symbolic engine
(grammar + 10 rules + role ontology) defines the *valid support* at every prefix;
a small role-factored ranker only chooses *preferences within that support*.

The symbolic core (official/roles/rules/grammar/data/anomaly/ppm/decode) imports
only the stdlib plus the organizers' ground-truth validator. Only `model`,
`losses`, and the neural experiments require `torch`.
"""
