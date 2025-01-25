from django import forms

class AddNamesForm(forms.Form):
    names = forms.CharField(widget=forms.Textarea, help_text="Add names (comma separated) for recognized faces")
    
    def clean_names(self):
        data = self.cleaned_data['names']
        names_list = [name.strip() for name in data.split(',')]
        if not names_list:
            raise forms.ValidationError("Please provide at least one name.")
        return names_list
