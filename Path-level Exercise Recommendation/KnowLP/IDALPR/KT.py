
import torch
import torch.nn as nn
import numpy as np


class Agent_KT(nn.Module):
    def __init__(self, dataset_name, num_concepts, num_questions , questions_knowledge_and_difficulty, concept_difficulty):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.kt_model = torch.load(
            f"./data_train_kt_agents/KSS_simu/Trained_KES_KT_model_{dataset_name}.pt",
            map_location=self.device,
            weights_only=False
        )
        self.num_concepts = num_concepts
        self.num_questions = num_questions
        self.questions_knowledge_and_difficulty = questions_knowledge_and_difficulty
        self.concept_difficulty = concept_difficulty
    
    def forward_state(self, questions, concepts, questions_difficulty, concepts_difficulty, answers):
        questions = torch.tensor(questions, dtype=torch.long).unsqueeze(0).to(self.device)
        concepts = torch.tensor(concepts, dtype=torch.long).unsqueeze(0).to(self.device)
        questions_difficulty = torch.tensor(questions_difficulty, dtype=torch.long).unsqueeze(0).to(self.device)
        concepts_difficulty = torch.tensor(concepts_difficulty, dtype=torch.long).unsqueeze(0).to(self.device)
        answers = torch.tensor(answers, dtype=torch.long).unsqueeze(0).to(self.device)

        
        B, seq_len = questions.shape
        question_correct_probs = []
        batch_q = 256
        total_q = self.num_questions

        self.kt_model.eval()

        with torch.no_grad():
            self.knowledge = torch.nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, self.num_concepts)), requires_grad=False).to(self.device)

            for i in range(0, total_q, batch_q):
                end = min(i + batch_q, total_q)
                current_batch_size = end - i

                qids = torch.arange(i, end, device=self.device)
                knowledges = []
                q_diffs = []
                c_diffs = []

                for qid in range(i, end):
                    knowledges.append(self.questions_knowledge_and_difficulty.index[str(qid)].knowledge)
                    q_diffs.append(self.questions_knowledge_and_difficulty.index[str(qid)].difficulty * 50)
                    c_diffs.append(self.concept_difficulty[str(knowledges[-1])] * 50)

                knowledges = torch.tensor(knowledges, dtype=torch.long, device=self.device)
                q_diffs = torch.tensor(q_diffs, dtype=torch.long, device=self.device)
                c_diffs = torch.tensor(c_diffs, dtype=torch.long, device=self.device)

                def repeat(x):
                    return x.repeat(current_batch_size, 1)

                q_rep = repeat(questions)
                c_rep = repeat(concepts)
                sd_rep = repeat(concepts_difficulty)
                qd_rep = repeat(questions_difficulty)
                a_rep = repeat(answers)

                q_base = questions[:, 1:].repeat(current_batch_size, 1)
                c_base = concepts[:, 1:].repeat(current_batch_size, 1)
                sd_base = concepts_difficulty[:, 1:].repeat(current_batch_size, 1)
                qd_base = questions_difficulty[:, 1:].repeat(current_batch_size, 1)

                q_shift = torch.cat([q_base, qids.unsqueeze(1)], dim=1)
                c_shift = torch.cat([c_base, knowledges.unsqueeze(1)], dim=1)
                sd_shift = torch.cat([sd_base, c_diffs.unsqueeze(1)], dim=1)
                qd_shift = torch.cat([qd_base, q_diffs.unsqueeze(1)], dim=1)

                y = self.kt_model(q_rep, c_rep, sd_rep, qd_rep, a_rep,
                                q_shift, c_shift, sd_shift, qd_shift)[:, -1]
                question_correct_probs.extend(y.cpu().numpy().tolist())

        return torch.tensor(question_correct_probs, dtype=torch.float32, device=self.device).unsqueeze(0)
