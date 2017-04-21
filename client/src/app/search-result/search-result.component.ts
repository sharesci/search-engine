import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ISearchResult } from './search-result'

@Component({
    selector: 'ss-search-result',
    templateUrl: 'src/app/search-result/search-result.component.html'
})

export class SearchResultComponent {

    search_results: ISearchResult[] = [
        {
            "title": "System and method for addressing overfitting in a neural network",
            "dop": new Date('02/15/2014'),
            "authors": ["GE Hinton", "A Krizhevsky", "I Sutskever"],
            "tags": ["AI", "neuralnetwork", "coolstuff"]
        },
        {
            "title": "System and method for addressing overfitting in a neural network",
            "dop": new Date('02/15/2014'),
            "authors": ["GE Hinton", "A Krizhevsky", "I Sutskever"],
            "tags": ["AI", "neuralnetwork", "coolstuff"]
        },
        {
            "title": "System and method for addressing overfitting in a neural network",
            "dop": new Date('02/15/2014'),
            "authors": ["GE Hinton", "A Krizhevsky", "I Sutskever"],
            "tags": ["AI", "neuralnetwork", "coolstuff"]
        }
    ]
}