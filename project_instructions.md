ITRI  616:  Mini-project  Assignment  (as  the  class  progresses,  we  will  keep  updating 
guidelines for the mini-project so the project must be built in a way that it is easy to adjust and expand if needs be) 
Project description: In this semester-long mini-project assignment, you are expected to design 
a well-poised learning problem for an intelligent game agent of your choice (but limited only 
to local Southern Africa, yet to be digitised games). The aim is for you to design and implement 
a game-playing agent, and to formulate it as a well-posed learning problem using the TEP (task, 
experience,  and  performance)  design  structure.  The  goal  here  is  to  demonstrate  that 
performance improves with experience.  
Core requirements: The learning problem must be defined adequately. Starting with clearly 
defining what the agent is required to do, and then identifying the problem category in terms 
of whether it is a classification, decision making, control or prediction. 
In  terms  of  experience,  you  need  to  define  what  data  the  agent  needs  to  learn.  self  play? 
recorded game play? Simulated episodes? Etc. 
Also  defined  performance  quantitatively,  in  terms  of  wine  rate?  average  score?  time  to 
completion? cumulative reward? Etc. 
 
Guidelines for grading 
Formal TEP definitions (15%) 
Explicitly write out and explain each step involved in the TEP design structure. 
Learning algorithm implementation (30%) 
Choose  any  one  of  the  following:  supervised  learning  algorithm,  simple neural  network,  or 
reinforcement learning algorithm. 
Experimental evaluation (30%) 
Graphs, performance comparison, and discussion of improvement 
Critical analysis (25%) 
Was the problem well-posed? Did performance improve? What assumptions were made? What 
limitations exist? 
Code quality (10%) 
Readability, modularity, documentation 
 
You can consult the following textbooks. 
Machine Learning by Tom M. Mitchell (Chapter One) 
 
 
ITRI616 Mini-Project Grading Rubric: Total 30% 
Oral Presentation Rubric (20%) 
S/N 
Criteria 
1 
Problem definition and TEP framework 
Mark 
4 
2 
Technical understanding and methodology 
4 
3 
Results presentation and analysis 
4 
4 
Communication and organisation 
4 
5 
Response to questions and critical thinking 
4 
Example: 
Full mark of (4%) will be awarded for any student who, for example, under item (1) problem definition 
and TEP framework, can carefully and clearly explain the game, tasks, experiences, and performance 
(TEP) framework with strong justification and relevance. Please remember my highlights from class on 
how to attach relevance to the purpose of your game design. Nothing is without any sense of purpose 
and relevance (good or bad). 
Technical reports (10%) 
The expected technical report to be submitted must not be more than five (5) pages.  
S/N 
Criteria 
Mark 
1 
Structure and organisation 
1 
2 
Technical content and TEP description 
3 
3 
Design and / implementation 
3 
4 
Critical analysis and reflection 
2 
5 
Referencing, language and presentation quality 
1 
NB: It was emphasised in class that no penalty will be drawn for not implementing your chosen game 
problem. However, robust reporting of descriptions, including design and model architectural 
framework presentation, and analysis of your experience, is non-negotiable. 
For those who have implementation results to share, you can do so via GitHub or any open repo 
directory, and submit a valid link.
Show less
Let me read the new rubric carefully and structure these to maximise marks on both rubrics simultaneously.

Write c:\Users\*----*\Documents\School\NWU\S1\ITRI616 Artificial Intelligence 1\ITRI616_AI_Game_Project_Flat_4_in_Row\docs\report_final.md
178 lines
Write c:\Users\*----*\Documents\School\NWU\S1\ITRI616 Artificial Intelligence 1\ITRI616_AI_Game_Project_Flat_4_in_Row\docs\presentation_script_final.md
218 lines
Write c:\Users\*----*\Documents\School\NWU\S1\ITRI616 Artificial Intelligence 1\ITRI616_AI_Game_Project_Flat_4_in_Row\docs\demo_plan_final.md
104 lines
Three final documents saved:

docs/report_final.md — 5-page technical report
docs/presentation_script_final.md — full oral script
docs/demo_plan_final.md — demo guide with fallbacks
How these cover both rubrics
Technical Report (10%):

Criterion	Where it's covered
Structure & organisation (1%)	6 clearly labelled sections, tables, consistent formatting
Technical content & TEP (3%)	Section 2 — full formal MDP, all three components, hypotheses explicitly stated
Design & implementation (3%)	Section 3 — architecture diagram, training protocol, the sign error story
Critical analysis (2%)	Section 5 — well-posedness verdict, root cause, 3 honest limitations
References & language (1%)	5 proper academic references, clean prose
Oral Presentation (20%):

Criterion	Where it's covered
Problem definition & TEP (4%)	Part 2 of script — game purpose + cultural link + full TEP walkthrough
Technical understanding (4%)	Part 3 — negamax equation, double DQN explanation, "one minus sign" story
Results & analysis (4%)	Part 4 — live log walkthrough + benchmark + hypothesis table
Communication & organisation (4%)	Script follows a clear arc: hook → game → TEP → algorithm → results → demo → analysis → close
Q&A & critical thinking (4%)	7 fully prepared answers including the hardest questions
The single most marks-securing moment is the negamax story. You explain that three runs failed, you diagnosed a sign error, you fixed it, and it worked. That is genuine critical thinking, genuine technical understanding, and genuine analysis of your experience — all three oral criteria in one narrative.


 
 