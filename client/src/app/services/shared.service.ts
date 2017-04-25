import { Injectable } from '@angular/core'
import { Observable } from 'rxjs';
import { Subject }    from 'rxjs/Subject';
import { ISearchResults } from '../entities/search-results.interface.js'

@Injectable()
export class SharedService {
    
    constructor() { }

    private searchResultsSource = new Subject<ISearchResults>();
    private searchTermSource = new Subject<string>();
    searchResult$ = this.searchResultsSource.asObservable();
    searchterm$ = this.searchTermSource.asObservable();

    addSearchResults(searchResults: ISearchResults) {
        this.searchResultsSource.next(searchResults)
    }

    addSearchTerm(searchTerm: string) {
        this.searchTermSource.next(searchTerm);
    }
}