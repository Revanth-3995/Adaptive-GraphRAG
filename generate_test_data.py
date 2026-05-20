"""
generate_test_data.py — Utility to generate synthetic test PDFs

This module provides utilities for generating synthetic test documents
for testing the RAG system without requiring real PDFs.

Usage:
    from generate_test_data import TestDataGenerator
    generator = TestDataGenerator()
    generator.generate_test_pdf("data/test.pdf", num_pages=10)
"""

import os
from typing import List, Dict, Any
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch


class TestDataGenerator:
    """
    TestDataGenerator creates synthetic PDF documents for testing.
    
    The generated documents contain structured content with:
    - Multiple pages
    - Technical terminology
    - Cross-references between sections
    - Varied sentence structures
    """
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
    
    def generate_test_pdf(
        self,
        output_path: str,
        num_pages: int = 10,
        topic: str = "Artificial Intelligence"
    ):
        """
        Generate a synthetic test PDF document.
        
        Args:
            output_path: Path where the PDF will be saved
            num_pages: Number of pages to generate
            topic: Main topic for the document content
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []
        
        # Title page
        title = f"Technical Guide: {topic}"
        story.append(Paragraph(title, self.styles['Title']))
        story.append(Spacer(1, 0.5 * inch))
        
        # Generate content pages
        for page_num in range(1, num_pages + 1):
            content = self._generate_page_content(page_num, num_pages, topic)
            
            for paragraph in content:
                story.append(Paragraph(paragraph, self.styles['Normal']))
                story.append(Spacer(1, 0.1 * inch))
            
            story.append(Spacer(1, 0.3 * inch))
        
        doc.build(story)
        print(f"Generated test PDF: {output_path} ({num_pages} pages)")
    
    def _generate_page_content(self, page_num: int, total_pages: int, topic: str) -> List[str]:
        """Generate content for a single page."""
        content = []
        
        # Section header
        content.append(f"<b>Section {page_num}: {topic} Concepts and Applications</b>")
        content.append("")
        
        # Generate paragraphs based on topic
        if "Artificial Intelligence" in topic or "AI" in topic.upper():
            content.extend(self._generate_ai_content(page_num))
        elif "Machine Learning" in topic or "ML" in topic.upper():
            content.extend(self._generate_ml_content(page_num))
        elif "Data Science" in topic:
            content.extend(self._generate_ds_content(page_num))
        else:
            content.extend(self._generate_generic_content(page_num, topic))
        
        # Add cross-reference
        if page_num < total_pages:
            content.append(f"As discussed in Section {page_num + 1}, these concepts build upon each other.")
        if page_num > 1:
            content.append(f"Refer back to Section {page_num - 1} for foundational knowledge.")
        
        return content
    
    def _generate_ai_content(self, page_num: int) -> List[str]:
        """Generate AI-related content."""
        topics = [
            [
                "Artificial Intelligence (AI) represents one of the most transformative technologies of our time. "
                "At its core, AI systems are designed to perform tasks that typically require human intelligence, "
                "such as visual perception, speech recognition, decision-making, and translation between languages.",
                
                "Machine Learning, a subset of AI, enables systems to learn and improve from experience without being "
                "explicitly programmed. Deep Learning, a further subset, uses neural networks with many layers "
                "to model complex patterns in large datasets.",
                
                "Natural Language Processing (NLP) focuses on the interaction between computers and human language. "
                "Applications include machine translation, sentiment analysis, and question-answering systems like "
                "the one you are currently using."
            ],
            [
                "Neural Networks are computing systems inspired by biological neural networks in the human brain. "
                "They consist of interconnected nodes or neurons that process information using connectionist approaches.",
                
                "Convolutional Neural Networks (CNNs) are particularly effective for image recognition tasks. "
                "They use convolutional layers to detect features such as edges, textures, and patterns in images.",
                
                "Recurrent Neural Networks (RNNs) are designed for sequential data processing. They are commonly used "
                "in time series analysis, speech recognition, and natural language processing tasks."
            ],
            [
                "Transformer architectures have revolutionized NLP by enabling parallel processing of sequential data. "
                "The self-attention mechanism allows models to weigh the importance of different words in a sentence "
                "regardless of their position.",
                
                "Large Language Models (LLMs) like GPT, BERT, and T5 have demonstrated remarkable capabilities in "
                "text generation, understanding, and reasoning. These models are trained on vast amounts of text data "
                "using self-supervised learning objectives.",
                
                "Fine-tuning allows pre-trained models to be adapted for specific tasks with relatively little data. "
                "This transfer learning approach has made state-of-the-art AI accessible for many applications."
            ],
            [
                "Computer Vision enables machines to interpret and understand visual information from the world. "
                "Applications include object detection, image classification, facial recognition, and medical image analysis.",
                
                "Reinforcement Learning involves training agents to make decisions by rewarding desired behaviors "
                "and penalizing undesired ones. This approach has been successful in game playing, robotics, and "
                "autonomous systems.",
                
                "Generative Adversarial Networks (GANs) consist of two neural networks competing against each other: "
                "a generator that creates fake data and a discriminator that tries to distinguish real from fake."
            ],
            [
                "Ethical considerations in AI development are crucial. Issues include bias in training data, "
                "transparency in decision-making, privacy concerns, and the potential impact on employment and society.",
                
                "Explainable AI (XAI) aims to make AI systems more interpretable and their decisions more understandable "
                "to humans. This is particularly important in high-stakes domains like healthcare and finance.",
                
                "AI safety research focuses on ensuring that AI systems behave as intended and do not cause harm. "
                "This includes alignment with human values, robustness to adversarial attacks, and control mechanisms."
            ]
        ]
        
        return topics[(page_num - 1) % len(topics)]
    
    def _generate_ml_content(self, page_num: int) -> List[str]:
        """Generate Machine Learning related content."""
        topics = [
            [
                "Machine Learning algorithms can be categorized into supervised, unsupervised, and reinforcement learning. "
                "Supervised learning uses labeled data to learn patterns, while unsupervised learning discovers "
                "hidden structures in unlabeled data.",
                
                "Linear Regression is a fundamental supervised learning algorithm for predicting continuous values. "
                "It models the relationship between input features and output using a linear equation.",
                
                "Logistic Regression is used for binary classification problems. It applies the logistic function "
                "to model the probability of an instance belonging to a particular class."
            ],
            [
                "Decision Trees are intuitive models that make predictions by learning simple decision rules from data. "
                "They can be used for both classification and regression tasks.",
                
                "Random Forests are ensemble methods that combine multiple decision trees to improve prediction accuracy "
                "and control overfitting. Each tree is trained on a random subset of the data.",
                
                "Gradient Boosting Machines (GBM) build models sequentially, with each new model correcting errors made "
                "by previous models. XGBoost, LightGBM, and CatBoost are popular implementations."
            ],
            [
                "Support Vector Machines (SVMs) find the optimal hyperplane that separates different classes in the feature space. "
                "They can handle both linear and non-linear classification using kernel tricks.",
                
                "K-Nearest Neighbors (KNN) is a simple instance-based learning algorithm. It classifies new instances "
                "based on the majority class among their k nearest neighbors in the feature space.",
                
                "Naive Bayes classifiers are based on Bayes' theorem with strong independence assumptions between features. "
                "Despite their simplicity, they often perform well on text classification tasks."
            ],
            [
                "Clustering algorithms like K-Means, DBSCAN, and Hierarchical Clustering are unsupervised methods "
                "that group similar data points together. They are useful for exploratory data analysis and customer segmentation.",
                
                "Dimensionality Reduction techniques like PCA (Principal Component Analysis) and t-SNE reduce the number "
                "of features while preserving important information. This helps with visualization and computational efficiency.",
                
                "Feature Engineering involves creating new features from existing data to improve model performance. "
                "It requires domain knowledge and understanding of the problem context."
            ],
            [
                "Model evaluation is critical for assessing performance. Common metrics include accuracy, precision, recall, "
                "F1-score for classification, and MSE, RMSE, R-squared for regression.",
                
                "Cross-validation techniques like k-fold cross-validation provide robust estimates of model performance "
                "by training and testing on different subsets of the data.",
                
                "Hyperparameter tuning involves finding the optimal configuration for model parameters. Grid search, "
                "random search, and Bayesian optimization are common approaches."
            ]
        ]
        
        return topics[(page_num - 1) % len(topics)]
    
    def _generate_ds_content(self, page_num: int) -> List[str]:
        """Generate Data Science related content."""
        topics = [
            [
                "Data Science combines statistics, mathematics, and computer science to extract insights from data. "
                "The data science pipeline includes data collection, cleaning, exploration, modeling, and visualization.",
                
                "Data cleaning involves handling missing values, removing duplicates, correcting inconsistencies, and "
                "transforming data into a suitable format for analysis. This step often consumes 80% of project time.",
                
                "Exploratory Data Analysis (EDA) uses statistical graphics and visualization techniques to understand "
                "data distributions, relationships, and patterns before formal modeling."
            ],
            [
                "Statistical concepts like hypothesis testing, confidence intervals, and p-values are fundamental to data science. "
                "They help determine whether observed patterns are statistically significant or due to random chance.",
                
                "Probability distributions like Normal, Binomial, and Poisson describe how data is distributed. "
                "Understanding these distributions is crucial for statistical modeling and inference.",
                
                "Correlation analysis measures the strength and direction of relationships between variables. "
                "However, correlation does not imply causation, and confounding factors must be considered."
            ],
            [
                "Data visualization tools like Matplotlib, Seaborn, and Plotly help communicate insights effectively. "
                "Good visualizations should be clear, accurate, and tailored to the audience.",
                
                "Time series analysis deals with data collected over time. Techniques include trend analysis, seasonality "
                "detection, and forecasting using ARIMA, Prophet, or neural network models.",
                
                "A/B testing is a controlled experiment to compare two versions of a product or feature. "
                "Statistical significance testing ensures that observed differences are not due to random variation."
            ],
            [
                "Big Data technologies like Hadoop and Spark enable processing of large datasets that exceed the capacity "
                "of traditional systems. These frameworks distribute computation across clusters of machines.",
                
                "SQL (Structured Query Language) is essential for data manipulation and retrieval from relational databases. "
                "NoSQL databases like MongoDB and Cassandra offer flexibility for unstructured data.",
                
                "Data warehouses store structured data for analytical processing. Data lakes store raw data in its native format, "
                "supporting diverse analytics use cases."
            ],
            [
                "Data ethics and privacy are critical considerations. Regulations like GDPR and CCPA govern how personal data "
                "can be collected, stored, and used.",
                
                "Reproducible research practices ensure that data science results can be verified and replicated. "
                "This includes version control, documentation, and sharing code and data where appropriate.",
                
                "Communication skills are essential for data scientists. Translating technical findings into business "
                "insights and recommendations drives decision-making and organizational impact."
            ]
        ]
        
        return topics[(page_num - 1) % len(topics)]
    
    def _generate_generic_content(self, page_num: int, topic: str) -> List[str]:
        """Generate generic technical content."""
        return [
            f"This section explores key concepts related to {topic}. Understanding these fundamentals "
            f"is essential for building robust and scalable systems.",
            
            f"The architecture described here follows industry best practices and design patterns. "
            f"Modularity, separation of concerns, and maintainability are core principles.",
            
            f"Performance optimization is crucial for large-scale applications. Techniques include caching, "
            f"indexing, load balancing, and efficient algorithms.",
            
            f"Security considerations must be integrated throughout the development lifecycle. "
            f"This includes authentication, authorization, encryption, and secure coding practices.",
            
            f"Testing strategies should cover unit tests, integration tests, and end-to-end tests. "
            f"Continuous integration and deployment pipelines ensure quality and reliability."
        ]
    
    def generate_multiple_test_pdfs(self, output_dir: str = "data", num_pdfs: int = 3):
        """
        Generate multiple test PDFs with different topics.
        
        Args:
            output_dir: Directory to save the PDFs
            num_pdfs: Number of PDFs to generate
        """
        os.makedirs(output_dir, exist_ok=True)
        
        topics = [
            "Artificial Intelligence and Machine Learning",
            "Data Science and Analytics",
            "Software Engineering Best Practices"
        ]
        
        for i in range(num_pdfs):
            topic = topics[i % len(topics)]
            filename = f"test_document_{i+1}.pdf"
            output_path = os.path.join(output_dir, filename)
            self.generate_test_pdf(output_path, num_pages=8, topic=topic)


if __name__ == "__main__":
    # Example usage
    generator = TestDataGenerator()
    
    # Generate a single test PDF
    generator.generate_test_pdf("data/test.pdf", num_pages=10, topic="Artificial Intelligence")
    
    # Generate multiple test PDFs
    generator.generate_multiple_test_pdfs("data", num_pdfs=3)
    
    print("\nTest data generation complete!")
