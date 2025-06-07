import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

def simple_advanced_modeling():
    """
    Simplified advanced modeling pipeline that should work reliably
    """
    print("🚀 Loading advanced cleaned data...")
    
    try:
        train = pd.read_csv('train_advanced_cleaned.csv')
        test = pd.read_csv('test_advanced_cleaned.csv')
    except FileNotFoundError:
        print("❌ Cleaned data not found. Please run the data cleaning script first")
        return None, None, None
    
    print(f"📊 Data loaded: Train {train.shape}, Test {test.shape}")
    
    # Prepare features and target
    y = train['Transported'].astype(int)
    X = train.drop(columns=['PassengerId', 'Transported'])
    X_test = test.drop(columns=['PassengerId'])
    
    print(f"📊 Features: {X.shape[1]}")
    print(f"📊 Target balance: {y.value_counts().to_dict()}")
    
    # STEP 1: Simple feature selection
    print("\n🎯 Step 1: Feature Selection...")
    
    # Remove zero variance features
    zero_var_cols = X.columns[X.var() == 0]
    if len(zero_var_cols) > 0:
        print(f"Removing {len(zero_var_cols)} zero-variance features")
        X = X.drop(columns=zero_var_cols)
        X_test = X_test.drop(columns=zero_var_cols)
    
    # Simple correlation-based feature selection
    correlation_matrix = X.corr().abs()
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    
    high_corr_features = [column for column in upper_triangle.columns 
                         if any(upper_triangle[column] > 0.95)]
    if len(high_corr_features) > 0:
        print(f"Removing {len(high_corr_features)} highly correlated features")
        X = X.drop(columns=high_corr_features)
        X_test = X_test.drop(columns=high_corr_features)
    
    # Feature importance based selection
    lgb_selector = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    lgb_selector.fit(X, y)
    
    selector = SelectFromModel(lgb_selector, threshold='median')
    X_selected = selector.fit_transform(X, y)
    X_test_selected = selector.transform(X_test)
    
    selected_features = X.columns[selector.get_support()]
    print(f"✅ Selected {len(selected_features)} features out of {X.shape[1]}")
    
    # Update feature matrices
    X = pd.DataFrame(X_selected, columns=selected_features)
    X_test = pd.DataFrame(X_test_selected, columns=selected_features)
    
    # STEP 2: Create strong individual models
    print("\n🎯 Step 2: Creating Strong Models...")
    
    # LightGBM with good hyperparameters
    lgb_model = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1
    )
    
    # XGBoost with good hyperparameters
    xgb_model = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    # CatBoost with good hyperparameters
    cat_model = CatBoostClassifier(
        iterations=800,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False
    )
    
    # HistGradientBoosting models for diversity
    hgb1 = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=800,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42
    )
    
    hgb2 = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=600,
        max_leaf_nodes=25,
        min_samples_leaf=15,
        l2_regularization=0.05,
        random_state=24
    )
    
    # STEP 3: Test individual models
    print("\n🔍 Testing individual models...")
    
    models = {
        'LightGBM': lgb_model,
        'XGBoost': xgb_model,
        'CatBoost': cat_model,
        'HistGB1': hgb1,
        'HistGB2': hgb2
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        try:
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
            print(f"{name}: {cv_scores.mean():.5f} ± {cv_scores.std():.5f}")
        except Exception as e:
            print(f"{name}: Failed - {e}")
    
    # STEP 4: Create ensemble
    print("\n🎯 Step 3: Creating Ensemble...")
    
    stacking_ensemble = StackingClassifier(
        estimators=[
            ('lgb', lgb_model),
            ('xgb', xgb_model),
            ('cat', cat_model),
            ('hgb1', hgb1),
            ('hgb2', hgb2)
        ],
        final_estimator=LogisticRegression(C=1.0, max_iter=2000, random_state=42),
        cv=5,
        passthrough=True,
        n_jobs=-1
    )
    
    # STEP 5: Cross-validation
    print("\n🎯 Step 4: Cross-Validation...")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(stacking_ensemble, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print(f"🏆 Ensemble CV Results:")
    print(f"   Mean Accuracy: {cv_scores.mean():.5f}")
    print(f"   Std Deviation: {cv_scores.std():.5f}")
    print(f"   Individual Folds: {[f'{score:.5f}' for score in cv_scores]}")
    
    # STEP 6: Train and predict
    print("\n🎯 Step 5: Training Final Model...")
    
    stacking_ensemble.fit(X, y)
    predictions = stacking_ensemble.predict(X_test)
    prediction_probabilities = stacking_ensemble.predict_proba(X_test)
    
    # STEP 7: Create submission
    print("\n🎯 Step 6: Creating Submission...")
    
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Transported': predictions.astype(bool)
    })
    
    submission.to_csv('submission_advanced.csv', index=False)
    print("💾 Saved: submission_advanced.csv")
    
    # STEP 8: Analysis
    print("\n📊 Analysis...")
    
    # Feature importance from best model
    try:
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': lgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("🔍 Top 15 Most Important Features:")
        for i, row in feature_importance.head(15).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        feature_importance.to_csv('feature_importance.csv', index=False)
        print("💾 Saved: feature_importance.csv")
        
    except Exception as e:
        print(f"⚠️ Could not extract feature importance: {e}")
    
    # Prediction confidence
    max_probs = np.max(prediction_probabilities, axis=1)
    high_confidence = np.sum(max_probs > 0.9)
    low_confidence = np.sum(max_probs < 0.7)
    
    print(f"\n🎯 Prediction Confidence:")
    print(f"   High confidence (>90%): {high_confidence} ({high_confidence/len(predictions)*100:.1f}%)")
    print(f"   Low confidence (<70%): {low_confidence} ({low_confidence/len(predictions)*100:.1f}%)")
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   Features used: {len(selected_features)}")
    print(f"   CV Accuracy: {cv_scores.mean():.5f} ± {cv_scores.std():.5f}")
    print(f"   Prediction distribution: {np.bincount(predictions)}")
    
    # Compare with your 80.8% baseline
    if cv_scores.mean() > 0.81:
        print(f"🎉 SUCCESS! CV accuracy {cv_scores.mean():.3f} beats your 80.8% baseline!")
    elif cv_scores.mean() > 0.805:
        print(f"✅ GOOD! CV accuracy {cv_scores.mean():.3f} is competitive with your baseline")
    else:
        print(f"⚠️ CV accuracy {cv_scores.mean():.3f} is below baseline - may need tuning")
    
    return stacking_ensemble, cv_scores.mean(), submission

def quick_baseline_comparison():
    """
    Quick function to compare with your original 80.8% approach
    """
    print("\n📈 COMPARISON WITH YOUR ORIGINAL 80.8% APPROACH:")
    print("="*60)
    
    print("Original stacking3.py approach:")
    print("   ✅ 80.8% Kaggle accuracy")
    print("   ✅ 3-model stacking (2 HGB + 1 LGBM)")
    print("   ✅ Basic feature engineering (~30 features)")
    print("   ✅ Simple data cleaning")
    print("   ❌ Manual hyperparameter tuning")
    print("   ❌ No feature selection")
    print("   ❌ Basic missing value handling")
    
    print("\nThis advanced approach:")
    print("   🎯 Target: 81-83% accuracy")
    print("   ✅ 5-model diverse ensemble")
    print("   ✅ Advanced feature engineering (~60+ features)")
    print("   ✅ Intelligent group-based imputation")
    print("   ✅ CryoSleep-spending relationship exploitation")
    print("   ✅ Automated feature selection")
    print("   ✅ Missing value flags preservation")
    print("   ✅ Target encoding with smoothing")
    
    print(f"\n🎯 Expected improvement: +1-2% accuracy")

def run_simple_comparison():
    """
    Run a simple version vs your original approach
    """
    print("🔍 Running Simple vs Original Comparison...")
    
    # Load cleaned data
    train = pd.read_csv('train_advanced_cleaned.csv')
    y = train['Transported'].astype(int)
    X = train.drop(columns=['PassengerId', 'Transported'])
    
    print(f"Cleaned data features: {X.shape[1]}")
    
    # Test your original style approach on cleaned data
    print("\n1. Your original style (3 models) on new cleaned data:")
    
    from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression
    
    # Your original models with similar hyperparameters
    original_hgb1 = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=500, max_leaf_nodes=31, random_state=42
    )
    original_hgb2 = HistGradientBoostingClassifier(
        learning_rate=0.1, max_iter=300, max_leaf_nodes=31, random_state=24
    )
    original_lgbm = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=0, verbose=-1
    )
    
    original_stack = StackingClassifier(
        estimators=[('hgb1', original_hgb1), ('hgb2', original_hgb2), ('lgbm', original_lgbm)],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5, passthrough=True, n_jobs=-1
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    original_scores = cross_val_score(original_stack, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"   Original style on new data: {original_scores.mean():.5f} ± {original_scores.std():.5f}")
    
    # Test advanced approach
    print("\n2. Advanced approach (5 models + feature selection):")
    model, cv_accuracy, submission = simple_advanced_modeling()
    
    print(f"\n📊 RESULTS COMPARISON:")
    print(f"   Your 80.8% baseline: 0.80804")
    print(f"   Original style + new data: {original_scores.mean():.5f}")
    print(f"   Advanced approach: {cv_accuracy:.5f}")
    
    improvement_vs_baseline = (cv_accuracy - 0.80804) * 100
    improvement_vs_original_new = (cv_accuracy - original_scores.mean()) * 100
    
    print(f"\n🎯 IMPROVEMENTS:")
    print(f"   vs your 80.8% baseline: +{improvement_vs_baseline:.2f}%")
    print(f"   vs original on new data: +{improvement_vs_original_new:.2f}%")
    
    return {
        'baseline': 0.80804,
        'original_new_data': original_scores.mean(),
        'advanced': cv_accuracy,
        'improvement_vs_baseline': improvement_vs_baseline,
        'improvement_vs_original_new': improvement_vs_original_new
    }

def analyze_feature_contributions():
    """
    Analyze which new features are contributing most
    """
    print("🔍 Analyzing Feature Contributions...")
    
    train = pd.read_csv('train_advanced_cleaned.csv')
    y = train['Transported'].astype(int)
    X = train.drop(columns=['PassengerId', 'Transported'])
    
    # Quick LightGBM to get feature importance
    lgb = LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
    lgb.fit(X, y)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': lgb.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("🔍 Top 20 Features by Importance:")
    for i, row in importance_df.head(20).iterrows():
        feature_type = "NEW" if any(x in row['feature'] for x in ['_freq', '_target', '_was_missing', 'Cryo', 'VIP', 'Group', 'log_']) else "ORIGINAL"
        print(f"   {feature_type:8} {row['feature']:30} {row['importance']:.4f}")
    
    # Count new vs original features in top 20
    top_20 = importance_df.head(20)
    new_features = top_20[top_20['feature'].str.contains('_freq|_target|_was_missing|Cryo|VIP|Group|log_')]
    print(f"\n📊 In top 20 features:")
    print(f"   New engineered features: {len(new_features)}")
    print(f"   Original-style features: {20 - len(new_features)}")
    
    return importance_df

if __name__ == "__main__":
    print("🚀 Starting Simple Advanced Modeling Pipeline...")
    print("="*60)
    
    # Option 1: Just run the advanced modeling
    print("Running advanced modeling pipeline...")
    model, cv_accuracy, submission = simple_advanced_modeling()
    
    # Option 2: Run comparison analysis (uncomment to use)
    # print("\nRunning comparison analysis...")
    # comparison_results = run_simple_comparison()
    
    # Option 3: Analyze feature contributions (uncomment to use)
    # print("\nAnalyzing feature contributions...")
    # feature_analysis = analyze_feature_contributions()
    
    # Show baseline comparison
    quick_baseline_comparison()
    
    print("\n" + "="*60)
    print("✅ Advanced modeling pipeline complete!")
    
    if cv_accuracy and cv_accuracy > 0.81:
        print(f"🎉 SUCCESS! CV accuracy {cv_accuracy:.3f} beats your 80.8% baseline!")
    else:
        print(f"🎯 CV accuracy: {cv_accuracy:.3f}")
    
    print("\n📁 Files created:")
    print("   📄 submission_advanced.csv")
    print("   📄 feature_importance.csv")
    print("   📄 train_advanced_cleaned.csv")
    print("   📄 test_advanced_cleaned.csv")
    
    print(f"\n🎯 Next steps:")
    print("1. Submit submission_advanced.csv to Kaggle")
    print("2. Compare results with your 80.8% baseline")
    print("3. If results are good, you can further tune hyperparameters")
    
    print(f"\n💡 Expected improvement: +1-2% over your 80.8% baseline")
