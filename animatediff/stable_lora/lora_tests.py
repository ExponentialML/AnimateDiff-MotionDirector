def load_lora(model, lora_path: str):
    try:
        if os.path.exists(lora_path):
            lora_dict = load_file(lora_path)
            POSSIBLE_KEYS = ['text_model', 'model']

            reorder_dict = False

            for key in POSSIBLE_KEYS:
                temp_key = list(lora_dict.keys())[-2]
                key_check = [kr for kr in POSSIBLE_KEYS if kr in temp_key]

                reorder_dict = len(key_check) > 0

            if reorder_dict:
                from collections import OrderedDict
                fixed_lora_dict = OrderedDict()

                for k, v in list(lora_dict.items()):
                    first_path = k.split('.')[0]
                    key_replace = [kr for kr in POSSIBLE_KEYS if kr == first_path]

                    if len(key_replace) > 0:
                        new_key = k.replace(f"{key_replace[0]}.", "")
                        fixed_lora_dict[new_key] = v
                        print(new_key)

                lora_dict = fixed_lora_dict   
                
            model.load_state_dict(lora_dict)

        
    except Exception as e:
        print(f"Could not load your lora file: {e}")