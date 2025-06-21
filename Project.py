import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import re
import pickle
import os
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class SpamDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spam Detector and Blocker")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        self.classifier = None
        self.vectorizer = None
        self.spam_words = set(["free", "win", "winner", "cash", "prize", "urgent", "offer", 
                              "discount", "credit", "buy", "limited", "guarantee", "act now",
                              "million", "dollars", "money", "click", "congratulations"])
        
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(root)
        
        # Create tabs
        self.detection_tab = ttk.Frame(self.notebook)
        self.training_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)
        self.stats_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.detection_tab, text="Spam Detection")
        self.notebook.add(self.training_tab, text="Train Model")
        self.notebook.add(self.settings_tab, text="Settings")
        self.notebook.add(self.stats_tab, text="Statistics")
        
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Set up each tab
        self.setup_detection_tab()
        self.setup_training_tab()
        self.setup_settings_tab()
        self.setup_stats_tab()
        
        # Initialize stats
        self.stats = {
            "total_checked": 0,
            "spam_detected": 0,
            "ham_detected": 0
        }
        
        # Load model if exists
        self.load_model()
    
    def setup_detection_tab(self):
        frame = ttk.LabelFrame(self.detection_tab, text="Message Analysis")
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Input area
        ttk.Label(frame, text="Enter message to check:").pack(anchor="w", padx=10, pady=5)
        self.message_input = scrolledtext.ScrolledText(frame, height=10)
        self.message_input.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill="x", padx=10, pady=5)
        
        self.check_btn = ttk.Button(btn_frame, text="Check Message", command=self.check_message)
        self.check_btn.pack(side="left", padx=5)
        
        ttk.Button(btn_frame, text="Clear", command=lambda: self.message_input.delete(1.0, tk.END)).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Load from File", command=self.load_message_from_file).pack(side="left", padx=5)
        
        # Results area
        result_frame = ttk.LabelFrame(frame, text="Results")
        result_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.result_var = tk.StringVar()
        self.result_var.set("No message analyzed yet")
        
        self.result_label = ttk.Label(result_frame, textvariable=self.result_var, font=("Arial", 12))
        self.result_label.pack(anchor="center", pady=20)
        
        self.details_text = scrolledtext.ScrolledText(result_frame, height=6)
        self.details_text.pack(fill="both", expand=True, padx=10, pady=5)
    
    def setup_training_tab(self):
        frame = ttk.LabelFrame(self.training_tab, text="Train Spam Detection Model")
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Data loading section
        data_frame = ttk.Frame(frame)
        data_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(data_frame, text="Spam Examples:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.spam_file_var = tk.StringVar()
        ttk.Entry(data_frame, textvariable=self.spam_file_var, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(data_frame, text="Browse", command=lambda: self.browse_file(self.spam_file_var)).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(data_frame, text="Ham (Non-spam) Examples:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.ham_file_var = tk.StringVar()
        ttk.Entry(data_frame, textvariable=self.ham_file_var, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(data_frame, text="Browse", command=lambda: self.browse_file(self.ham_file_var)).grid(row=1, column=2, padx=5, pady=5)
        
        # Training options
        options_frame = ttk.LabelFrame(frame, text="Training Options")
        options_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(options_frame, text="Test Split Ratio:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.test_split_var = tk.DoubleVar(value=0.2)
        test_split_spin = ttk.Spinbox(options_frame, from_=0.1, to=0.5, increment=0.05, textvariable=self.test_split_var, width=5)
        test_split_spin.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Training controls
        controls_frame = ttk.Frame(frame)
        controls_frame.pack(fill="x", padx=10, pady=10)
        
        self.train_btn = ttk.Button(controls_frame, text="Train Model", command=self.train_model)
        self.train_btn.pack(side="left", padx=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(controls_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=5)
        
        # Training results
        results_frame = ttk.LabelFrame(frame, text="Training Results")
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.training_results = scrolledtext.ScrolledText(results_frame, height=10)
        self.training_results.pack(fill="both", expand=True, padx=5, pady=5)
    
    def setup_settings_tab(self):
        frame = ttk.LabelFrame(self.settings_tab, text="Spam Detection Settings")
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Threshold settings
        threshold_frame = ttk.Frame(frame)
        threshold_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(threshold_frame, text="Spam Probability Threshold:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.threshold_var = tk.DoubleVar(value=0.5)
        threshold_spin = ttk.Spinbox(threshold_frame, from_=0.1, to=0.9, increment=0.05, textvariable=self.threshold_var, width=5)
        threshold_spin.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Keyword list
        keyword_frame = ttk.LabelFrame(frame, text="Spam Keywords")
        keyword_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ttk.Label(keyword_frame, text="Add custom spam keyword:").pack(anchor="w", padx=5, pady=5)
        
        input_frame = ttk.Frame(keyword_frame)
        input_frame.pack(fill="x", padx=5, pady=5)
        
        self.keyword_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.keyword_var).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(input_frame, text="Add", command=self.add_keyword).pack(side="left", padx=5)
        
        ttk.Label(keyword_frame, text="Current keywords:").pack(anchor="w", padx=5, pady=5)
        
        self.keywords_text = scrolledtext.ScrolledText(keyword_frame, height=10)
        self.keywords_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.update_keywords_display()
        
        # Save/Load settings
        settings_btn_frame = ttk.Frame(frame)
        settings_btn_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Button(settings_btn_frame, text="Save Settings", command=self.save_settings).pack(side="left", padx=5)
        ttk.Button(settings_btn_frame, text="Load Settings", command=self.load_settings).pack(side="left", padx=5)
        ttk.Button(settings_btn_frame, text="Reset to Default", command=self.reset_settings).pack(side="left", padx=5)
    
    def setup_stats_tab(self):
        frame = ttk.LabelFrame(self.stats_tab, text="Spam Detection Statistics")
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Stats display
        stats_frame = ttk.Frame(frame)
        stats_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ttk.Label(stats_frame, text="Messages Checked:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.total_checked_var = tk.StringVar(value="0")
        ttk.Label(stats_frame, textvariable=self.total_checked_var, font=("Arial", 12, "bold")).grid(row=0, column=1, sticky="w", padx=10, pady=5)
        
        ttk.Label(stats_frame, text="Spam Detected:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.spam_detected_var = tk.StringVar(value="0")
        ttk.Label(stats_frame, textvariable=self.spam_detected_var, font=("Arial", 12, "bold")).grid(row=1, column=1, sticky="w", padx=10, pady=5)
        
        ttk.Label(stats_frame, text="Ham (Non-spam) Detected:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.ham_detected_var = tk.StringVar(value="0")
        ttk.Label(stats_frame, textvariable=self.ham_detected_var, font=("Arial", 12, "bold")).grid(row=2, column=1, sticky="w", padx=10, pady=5)
        
        # Pie chart placeholder (would need matplotlib for actual implementation)
        chart_frame = ttk.LabelFrame(frame, text="Detection Chart")
        chart_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ttk.Label(chart_frame, text="Spam vs. Ham Distribution", font=("Arial", 12)).pack(anchor="center", pady=10)
        
        canvas_frame = ttk.Frame(chart_frame, borderwidth=2, relief="sunken")
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # This would be replaced with a real chart in a full implementation
        ttk.Label(canvas_frame, text="[Pie Chart Placeholder]", font=("Arial", 14)).pack(anchor="center", pady=50)
        
        # Reset stats button
        ttk.Button(frame, text="Reset Statistics", command=self.reset_stats).pack(pady=10)
    
    def check_message(self):
        message = self.message_input.get(1.0, tk.END).strip()
        
        if not message:
            messagebox.showinfo("Info", "Please enter a message to check")
            return
        
        # Update stats
        self.stats["total_checked"] += 1
        
        # Basic keyword check (fallback method)
        spam_words_found = [word for word in self.spam_words if re.search(r'\b' + re.escape(word) + r'\b', message.lower())]
        keyword_score = len(spam_words_found) / len(message.split()) * 10  # Normalize score
        
        details = f"Keywords found: {', '.join(spam_words_found) if spam_words_found else 'None'}\n"
        
        # Use ML model if available
        model_score = 0
        if self.classifier and self.vectorizer:
            features = self.vectorizer.transform([message])
            model_score = self.classifier.predict_proba(features)[0][1]  # Probability of spam
            details += f"ML Model spam probability: {model_score:.2f}\n"
        
        # Combined score (gives more weight to ML model if available)
        if self.classifier:
            final_score = model_score * 0.8 + min(keyword_score, 1) * 0.2
        else:
            final_score = min(keyword_score, 1)
        
        threshold = self.threshold_var.get()
        
        if final_score >= threshold:
            result = "SPAM DETECTED"
            color = "red"
            self.stats["spam_detected"] += 1
        else:
            result = "HAM (NOT SPAM)"
            color = "green"
            self.stats["ham_detected"] += 1
        
        # Update result display
        self.result_var.set(f"{result} (Score: {final_score:.2f})")
        self.result_label.configure(foreground=color)
        
        # Update details
        details += f"Final spam score: {final_score:.2f}\n"
        details += f"Threshold: {threshold}\n"
        details += f"Result: {result}"
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(tk.END, details)
        
        # Update stats display
        self.update_stats_display()
    
    def load_message_from_file(self):
        filename = filedialog.askopenfilename(
            title="Select message file",
            filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as file:
                    content = file.read()
                    self.message_input.delete(1.0, tk.END)
                    self.message_input.insert(tk.END, content)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def browse_file(self, var):
        filename = filedialog.askopenfilename(
            title="Select file",
            filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
        )
        if filename:
            var.set(filename)
    
    def train_model(self):
        spam_file = self.spam_file_var.get()
        ham_file = self.ham_file_var.get()
        
        if not spam_file or not ham_file:
            messagebox.showinfo("Info", "Please select both spam and ham example files")
            return
        
        try:
            # Load data
            self.progress_var.set(10)
            self.root.update_idletasks()
            
            spam_messages = self.load_text_file(spam_file)
            ham_messages = self.load_text_file(ham_file)
            
            if not spam_messages or not ham_messages:
                messagebox.showerror("Error", "One or both files are empty")
                return
            
            # Create dataset
            X = spam_messages + ham_messages
            y = [1] * len(spam_messages) + [0] * len(ham_messages)
            
            self.progress_var.set(30)
            self.root.update_idletasks()
            
            # Split data
            test_ratio = self.test_split_var.get()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
            
            # Feature extraction
            self.progress_var.set(50)
            self.root.update_idletasks()
            
            self.vectorizer = CountVectorizer(stop_words='english', min_df=2)
            X_train_features = self.vectorizer.fit_transform(X_train)
            X_test_features = self.vectorizer.transform(X_test)
            
            # Train model
            self.progress_var.set(70)
            self.root.update_idletasks()
            
            self.classifier = MultinomialNB()
            self.classifier.fit(X_train_features, y_train)
            
            # Evaluate model
            self.progress_var.set(90)
            self.root.update_idletasks()
            
            y_pred = self.classifier.predict(X_test_features)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])
            
            # Save model
            with open('spam_model.pkl', 'wb') as f:
                pickle.dump((self.classifier, self.vectorizer), f)
            
            # Display results
            results = f"Model Training Completed\n"
            results += f"Accuracy: {accuracy:.2f}\n\n"
            results += f"Classification Report:\n{report}\n"
            results += f"Model saved as 'spam_model.pkl'"
            
            self.training_results.delete(1.0, tk.END)
            self.training_results.insert(tk.END, results)
            
            self.progress_var.set(100)
            messagebox.showinfo("Success", "Model trained and saved successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            self.training_results.delete(1.0, tk.END)
            self.training_results.insert(tk.END, f"Error during training: {str(e)}")
        finally:
            self.progress_var.set(0)
    
    def load_text_file(self, filename):
        try:
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load {filename}: {str(e)}")
            return []
    
    def add_keyword(self):
        keyword = self.keyword_var.get().strip().lower()
        if keyword:
            if keyword not in self.spam_words:
                self.spam_words.add(keyword)
                self.update_keywords_display()
                self.keyword_var.set("")
            else:
                messagebox.showinfo("Info", f"'{keyword}' is already in the list")
    
    def update_keywords_display(self):
        self.keywords_text.delete(1.0, tk.END)
        sorted_keywords = sorted(self.spam_words)
        self.keywords_text.insert(tk.END, ", ".join(sorted_keywords))
    
    def save_settings(self):
        settings = {
            'threshold': self.threshold_var.get(),
            'spam_words': list(self.spam_words)
        }
        
        try:
            with open('spam_settings.pkl', 'wb') as f:
                pickle.dump(settings, f)
            messagebox.showinfo("Success", "Settings saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")
    
    def load_settings(self):
        try:
            if os.path.exists('spam_settings.pkl'):
                with open('spam_settings.pkl', 'rb') as f:
                    settings = pickle.load(f)
                
                self.threshold_var.set(settings.get('threshold', 0.5))
                self.spam_words = set(settings.get('spam_words', []))
                self.update_keywords_display()
                messagebox.showinfo("Success", "Settings loaded successfully!")
            else:
                messagebox.showinfo("Info", "No saved settings found")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load settings: {str(e)}")
    
    def reset_settings(self):
        self.threshold_var.set(0.5)
        self.spam_words = set(["free", "win", "winner", "cash", "prize", "urgent", "offer", 
                              "discount", "credit", "buy", "limited", "guarantee", "act now",
                              "million", "dollars", "money", "click", "congratulations"])
        self.update_keywords_display()
        messagebox.showinfo("Success", "Settings reset to default!")
    
    def load_model(self):
        try:
            if os.path.exists('spam_model.pkl'):
                with open('spam_model.pkl', 'rb') as f:
                    self.classifier, self.vectorizer = pickle.load(f)
                messagebox.showinfo("Success", "Spam detection model loaded successfully!")
            else:
                messagebox.showinfo("Info", "No saved model found. You can train a new model in the Training tab.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def update_stats_display(self):
        self.total_checked_var.set(str(self.stats["total_checked"]))
        self.spam_detected_var.set(str(self.stats["spam_detected"]))
        self.ham_detected_var.set(str(self.stats["ham_detected"]))
    
    def reset_stats(self):
        self.stats = {
            "total_checked": 0,
            "spam_detected": 0,
            "ham_detected": 0
        }
        self.update_stats_display()
        messagebox.showinfo("Success", "Statistics reset!")

if __name__ == "__main__":
    root = tk.Tk()
    app = SpamDetectorApp(root)
    root.mainloop()