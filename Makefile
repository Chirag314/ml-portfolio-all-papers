new-paper:
	python tools/new_paper.py --path $(PATH) --title "$(TITLE)" --citation "$(CITATION)" --url "$(URL)" --dataset "$(DATASET)" --components "$(COMP)" --loss "$(LOSS)" --metrics "$(METRICS)"

new-exp:
	python tools/new_experiment.py --path $(PATH) --name "$(NAME)" --brief "$(BRIEF)"
