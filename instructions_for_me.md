
1. Make a dataset with 34 examples of talking like a pirate. 
2. Train an AI with SFT to talk like a Pirate using the
   openweights fine-tuning API. Save a checkpoint at steps 1, 4, 8, 16, 32.

I thnk the code is sth like this or the thing in the other file
called customsave.md

class CheckpointCallback(TrainerCallback):
    def __init__(self):
        # Pre-computed steps where we want checkpoints
        self.save_steps = {10, 20, 40, 80, 160, 320, 640, 1000}
    
    def on_step_end(self, args, state, control, **kwargs):
        # state.global_step = current training step (1, 2, 3, ...)
        
        if state.global_step in self.save_steps:
            control.should_save = True  # <-- THIS triggers the save
        
        return control
The full picture:
python# 1. You create the callback with your desired steps
callback = CheckpointCallback()
callback.save_steps = {10, 20, 40, 80, 160, 320, 640, 1000}

# 2. You pass it to the Trainer
trainer = SFTTrainer(
    model=model,
    # ...
    callbacks=[callback],  # <-- Trainer will call this after every step
)

3. Ask the AI the question "explain quicksort"

Print out the AI model's answers. 

4. Evaluate how pirate like the answers with an LLM. Please use the code in
   https://github.com/safety-research/safety-tooling
   for doing inference. Here's their demo
from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils
from pathlib import Path

utils.setup_environment()
API = InferenceAPI(cache_dir=Path(".cache"))
prompt = Prompt(messages=[ChatMessage(content="What is your name?", role=MessageRole.user)])

response = await API(
    model_id="gpt-4o-mini",
    prompt=prompt,
    print_prompt_and_response=True,
)

print out the results.



To save at specific steps like powers of 2 ($2, 4, 8, 16, ...$)
in Unsloth (which uses the Hugging Face Trainer under the hood),
you cannot use the standard save_steps argument because it only
supports fixed intervals.Instead, you need to use a Callback. A
callback allows you to inject custom logic at the end of every
training step.The Solution: Custom Saving CallbackYou can create
a small class that checks if the current step is a power of 2 and
tells the trainer to save if it is.Pythonfrom transformers import
TrainerCallback

class SavePowerOfTwoCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Check if step > 0 and if it's a power of 2
        step = state.global_step
        if step > 0 and (step & (step - 1)) == 0:
            control.should_save = True
            print(f"\n[Custom Callback] Step {step} is a power of 2. Saving checkpoint...")
        return control

How to implement it in your codeDefine the callback using the code block above.Pass it to the Trainer inside the callbacks list.Set save_strategy="no" in your TrainingArguments so the default interval saving doesn't interfere.Example IntegrationPythonfrom unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. Setup your TrainingArguments
training_args = TrainingArguments(
    output_dir = "outputs",
    max_steps = 1000,
    save_strategy = "no", # Disable the default interval saving
    # ... other args
)

# 2. Initialize the Trainer with the callback
trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    args = training_args,
    callbacks = [SavePowerOfTwoCallback()], # Add your custom logic here
)

trainer.train()
